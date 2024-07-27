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
import optuna

# 再現性のためのランダムシード固定
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(trial, config):

    # 学習率と重み減衰の値を取得
    lr = round(trial.suggest_loguniform('lr', 1e-5, 1e-1), 6)
    weight_decay = round(trial.suggest_loguniform('weight_decay', 1e-10, 1e-3), 6)

    # ロガーの設定
    logger = config.get_logger('train')
    # trainデータローダーとvalidデータローダーの設定
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.split_validation()

    # モデルの定義
    model = feature_extractor
    # モデル情報の出力
    logger.info(model)

    # デバイスの設定
    device, device_ids = prepare_device(config['n_gpu'])
    # モデルをデバイスに転送
    model = model.to(device)

    # 複数GPUを使用する場合
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # 損失関数の定義と評価指標の設定
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # 学習パラメータの設定
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    # 最適化アルゴリズムの設定
    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)

    # トレーナーの設定
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=None)

    # トレーナーのトレーニングメソッドを呼び出し、検証セットでの最終ロスを返す
    loss = trainer.train()

    # 最も良かったエポックの時の検証損失を返す
    return loss

def objective(trial):
    # ConfigParserのインスタンスを作成
    args = argparse.ArgumentParser(description='Weather Forecasting')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    config = ConfigParser.from_args(args)
    return main(trial, config)

if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Best trial:")
    trial = study.best_trial
    print("    Value: ", trial.value)
    print("    Params: ")
    for key, value in trial.params.items():
        print(f"      {key}: {value}")
