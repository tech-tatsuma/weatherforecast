import torch
from abc import abstractmethod
from numpy import inf
from logger import TensorboardWriter


class BaseTrainer:
    """
    すべてのトレーナーの基底クラス
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config):
        # 設定ファイル
        self.config = config
        # ログ出力用のインスタンス
        self.logger = config.get_logger('trainer', config['trainer']['verbosity'])

        self.model = model # モデル
        self.criterion = criterion # 損失関数
        self.metric_ftns = metric_ftns # 評価指標の関数リスト
        self.optimizer = optimizer # 最適化アルゴリズム

        cfg_trainer = config['trainer']
        self.epochs = cfg_trainer['epochs'] # 総エポック数
        self.save_period = cfg_trainer['save_period'] # 保存間隔
        self.monitor = cfg_trainer.get('monitor', 'off') # モデルの監視設定

        # モデル性能の監視とベストモデルの保存の設定
        if self.monitor == 'off':
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.monitor.split()
            assert self.mnt_mode in ['min', 'max'] # 監視モードは'min'または'max'

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf # 監視する最良の値を初期設定
            self.early_stop = cfg_trainer.get('early_stop', inf) # 早期終了の閾値
            if self.early_stop <= 0:
                self.early_stop = inf # 早期終了が設定されていない場合

        self.start_epoch = 1 # 開始エポック

        self.checkpoint_dir = config.save_dir # チェックポイントの保存ディレクトリ

        # Tensorboardでの可視化ライターインスタンスの設定             
        self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        if config.resume is not None:
            self._resume_checkpoint(config.resume) # チェックポイントからの再開

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        エポックごとのトレーニングロジック

        :param epoch: 現在のエポック数
        """
        raise NotImplementedError # サブクラスで実装が必要

    def train(self):
        """
        トレーニングの全体的なロジック
        """
        not_improved_count = 0 # 改善されなかった回数のカウント
        best_result = None
        for epoch in range(self.start_epoch, self.epochs + 1):
            # エポックごとのトレーニングを実行
            result, model = self._train_epoch(epoch) # エポックごとのトレーニング実行

            # ログ情報を辞書に保存
            log = {'epoch': epoch}
            log.update(result)

            # モデルのパフォーマンス評価と最良のチェックポイントの保存
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))

            # モデルのパフォーマンス評価と最良のチェックポイントの保存
            best = False # このエポックで最良のモデルを更新したかどうかを示すフラグ
            if self.mnt_mode != 'off': # モデルのパフォーマンス監視が有効な場合
                try:
                    # モデルのパフォーマンスが前回の最良から改善されたかどうかをチェック
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                except KeyError:
                    # 指定されたメトリックがログに存在しない場合，警告を出力し監視を無効にする
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                # 改善が確認された場合，最良のスコアを更新し，改善されなかったカウントをリセット
                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                    best_result = log[self.mnt_metric]
                    best_model = model
                else:
                    # 改善されなかった場合，カウンターをインクリメント
                    not_improved_count += 1

                # 早期終了の条件を満たした場合，トレーニングを停止
                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

        arch = type(self.model).__name__
        state = {
            'arch': arch, # モデルのアーキテクチャ名
            'epoch': epoch, # 現在のエポック数
            'state_dict': best_model.state_dict(), # モデルの状態
            'optimizer': self.optimizer.state_dict(), # オプティマイザの状態
            'monitor_best': self.mnt_best, # 監視している最良の評価値
            'config': self.config # トレーニングの設定
        }
        torch.save(state, str(self.checkpoint_dir / 'best.pth'))
        # torch.save(state, "best_xavier.pth")
        
        return best_result

    def _resume_checkpoint(self, resume_path):
        """
        保存されたチェックポイントから再開

        :param resume_path: 再開するチェックポイントのパス
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path)) # チェックポイントの読み込み開始をログに出力
        checkpoint = torch.load(resume_path) # チェックポイントファイルを読み込み
        self.start_epoch = checkpoint['epoch'] + 1 # 次のエポックを設定
        self.mnt_best = checkpoint['monitor_best'] # 最良の評価値を更新

        # チェックポイントからアーキテクチャのパラメータをロード
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict']) # モデルの状態をロード

        # チェックポイントからオプティマイザの状態をロード（タイプが変わっていない場合）
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch)) # ロード完了をログに出力
