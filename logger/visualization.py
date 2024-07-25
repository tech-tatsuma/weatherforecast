import importlib
from datetime import datetime


class TensorboardWriter():
    def __init__(self, log_dir, logger, enabled):
        self.writer = None # TensorBoardのライターインスタンスを保持するための変数
        self.selected_module = "" # 使用するTensorBoardモジュールの名前を保持する変数

        if enabled:
            log_dir = str(log_dir) # ログディレクトリのパスを文字列に変換

            # TensorBoardのライターモジュールを見つけて初期化する
            succeeded = False
            for module in ["torch.utils.tensorboard", "tensorboardX"]:
                try:
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                    succeeded = True
                    break
                except ImportError:
                    succeeded = False
                self.selected_module = module # 成功したモジュールの名前を保持

            if not succeeded:
                # 必要なモジュールがインストールされていない場合の警告メッセージ
                message = "Warning: visualization (Tensorboard) is configured to use, but currently not installed on " \
                    "this machine. Please install TensorboardX with 'pip install tensorboardx', upgrade PyTorch to " \
                    "version >= 1.1 to use 'torch.utils.tensorboard' or turn off the option in the 'config.json' file."
                logger.warning(message)

        self.step = 0 # ステップカウンタ
        self.mode = '' # 現在のモード

        # TensorBoardに追加可能なデータの種類
        self.tb_writer_ftns = {
            'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio',
            'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'
        }
        # 特定の関数でモードをタグに追加しない例外
        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}
        self.timer = datetime.now() # タイマーの初期設定

    def set_step(self, step, mode='train'):
        self.mode = mode # 現在のモードを設定
        self.step = step # ステップ数を更新
        if step == 0:
            self.timer = datetime.now() # タイマーをリセット
        else:
            duration = datetime.now() - self.timer # 前回の呼び出しからの時間を計測
            self.add_scalar('steps_per_sec', 1 / duration.total_seconds()) # ステップの実行速度を記録
            self.timer = datetime.now() # タイマーを更新

    def __getattr__(self, name):
        """
        TensorBoardの使用が有効な場合、追加の情報（ステップ、タグ）を付加したデータ追加メソッドを返す。
        それ以外の場合は、何もしないダミー関数を返す。
        """
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)

            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    # モードをタグに追加(例外リストにない場合)
                    if name not in self.tag_mode_exceptions:
                        tag = '{}/{}'.format(tag, self.mode)
                    add_data(tag, data, self.step, *args, **kwargs) # データを追加
            return wrapper
        else:
            # このクラスで定義されているメソッドを返すか、属性エラーを発生させる
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object '{}' has no attribute '{}'".format(self.selected_module, name))
            return attr
