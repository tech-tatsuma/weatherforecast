import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
from model.model import feature_extractor


# 再現性のためのランダムシード固定
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True # 計算の再現性を保証
torch.backends.cudnn.benchmark = False # 性能向上のための最適化を無効化
np.random.seed(SEED) # NumPyのランダムシードを固定

def main(config):
    # トレーニング用のロガーを取得
    logger = config.get_logger('train')

    # データローダーインスタンスの設定
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation() # バリデーション用データローダの分割

    # モデルアーキテクチャの構築とコンソールへの出力
    model = feature_extractor
    logger.info(model)

    # GPUトレーニングの準備（複数デバイス対応）
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # 損失関数とメトリクス関数のハンドル取得
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # オプティマイザと学習率スケジューラの構築
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    # トレーナークラスのインスタンス作成
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=None)

    # トレーニング実行
    _ = trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Weather Forecasting')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # JSON設定ファイルからデフォルト値を変更するためのカスタムCLIオプション
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
