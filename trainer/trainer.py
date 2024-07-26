import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    トレーニングを管理するクラス
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)

        # 設定情報
        self.config = config
        # 使用するデバイス
        self.device = device
        # トレーニングデータローダー
        self.data_loader = data_loader

        if len_epoch is None:
            # エポック単位でのトレーニング
            self.len_epoch = len(self.data_loader)
        else:
            # イテレーション単位でのトレーニング（無限ループ）
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader # 検証データローダー
        self.do_validation = self.valid_data_loader is not None # 検証を行うかどうか
        self.lr_scheduler = lr_scheduler # 学習率スケジューラー
        self.log_step = int(np.sqrt(data_loader.batch_size)) # ログを記録するステップ間隔

        # トレーニングと検証のメトリクス管理
        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        1エポックのトレーニングロジック

        :param epoch: 現在のトレーニングエポック
        :return: このエポックの平均損失とメトリクスのログ
        """
        self.model.train() # モデルをトレーニングモードに設定
        self.train_metrics.reset() # メトリクスのリセット

        # バッチループ
        for batch_idx, (data, target, seq_x_mark) in enumerate(self.data_loader):
            # データをデバイスに転送
            data, target, seq_x_mark = data.to(self.device), target.to(self.device), seq_x_mark.to(self.device)

            self.optimizer.zero_grad() # 勾配をリセット
            output = self.model(data, seq_x_mark) # モデルで推論
            loss = self.criterion(output, target) # 損失計算
            loss.backward() # 勾配の計算
            self.optimizer.step() # パラメータ更新

            # ロギング
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log, self.model

    def _valid_epoch(self, epoch):
        """
        1エポックの検証ロジック

        :param epoch: 現在のトレーニングエポック
        :return: 検証の情報を含むログ
        """
        self.model.eval() # モデルを評価モードに設定
        self.valid_metrics.reset() # メトリクスのリセット
        with torch.no_grad(): # 勾配計算を無効化
            for batch_idx, (data, target, seq_x_mark) in enumerate(self.valid_data_loader):
                data, target, seq_x_mark = data.to(self.device), target.to(self.device), seq_x_mark.to(self.device)

                output = self.model(data, seq_x_mark)
                loss = self.criterion(output, target)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))

        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        """
        進行状況のフォーマット

        :param batch_idx: 現在のバッチインデックス
        :return: 進行状況の文字列
        """
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
