import os
import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
from logger import setup_logging
from utils import read_json, write_json


class ConfigParser:
    def __init__(self, config, resume=None, modification=None, run_id=None):
        """
        設定ファイルを解析するクラス。トレーニングのハイパーパラメータ、モジュールの初期化、チェックポイントの保存、
        ログモジュールの設定を扱います。
        :param config: トレーニングの設定とハイパーパラメータが含まれる辞書。例えば`config.json`の内容。
        :param resume: チェックポイントのパス。
        :param modification: 設定辞書から置換するためのキーチェーンと値の辞書。
        :param run_id: トレーニングプロセスのユニークID。チェックポイントとトレーニングログの保存に使用。デフォルトはタイムスタンプ。
        """
        # 設定ファイルをロードして修正を適用
        self._config = _update_config(config, modification)
        self.resume = resume

        # トレーニングモデルとログを保存するディレクトリを設定
        save_dir = Path(self.config['trainer']['save_dir'])

        exper_name = self.config['name']
        if run_id is None: # デフォルトのrun-idとしてタイムスタンプを使用
            run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        self._save_dir = save_dir / 'models' / exper_name / run_id
        self._log_dir = save_dir / 'log' / exper_name / run_id

        # チェックポイントログを保存するディレクトリを作成
        exist_ok = run_id == ''
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)

        # 更新された設定ファイルをチェックポイントディレクトリに保存
        write_json(self.config, self.save_dir / 'config.json')

        # ログモジュールを設定
        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    @classmethod
    def from_args(cls, args, options=''):
        """
        CLI引数からこのクラスを初期化します。トレーニング、テストに使用されます。
        """
        for opt in options:
            args.add_argument(*opt.flags, default=None, type=opt.type)
        if not isinstance(args, tuple):
            args = args.parse_args()

        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        if args.resume is not None:
            resume = Path(args.resume)
            cfg_fname = resume.parent / 'config.json'
        else:
            msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
            assert args.config is not None, msg_no_cfg
            resume = None
            cfg_fname = Path(args.config)
        
        config = read_json(cfg_fname)
        if args.config and resume:
            # 新しい設定で微調整するために新しい設定を更新
            config.update(read_json(args.config))

        # CLIオプションを辞書に解析
        modification = {opt.target : getattr(args, _get_opt_name(opt.flags)) for opt in options}
        return cls(config, resume, modification)

    def init_obj(self, name, module, *args, **kwargs):
        """
        設定で指定された'type'の名前で関数ハンドルを見つけ、対応する引数で初期化されたインスタンスを返します。
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        設定で指定された'type'の名前で関数ハンドルを見つけ、引数が固定された関数をpartialを使って返します。
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """通常の辞書のようにアイテムにアクセスすることができます。"""
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        """ログレベルに基づいてロガーを取得します。"""
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity, self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # 読み取り専用属性の設定
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

# 設定辞書をカスタムCLIオプションで更新するためのヘルパー関数
def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        if v is not None:
            _set_by_path(config, k, v)
    return config

def _get_opt_name(flags):
    """フラグからオプション名を取得します。"""
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')

def _set_by_path(tree, keys, value):
    """キーのシーケンスによってツリー内のネストされたオブジェクトに値を設定します。"""
    keys = keys.split(';')
    _get_by_path(tree, keys[:-1])[keys[-1]] = value

def _get_by_path(tree, keys):
    """キーのシーケンスによってツリー内のネストされたオブジェクトにアクセスします。"""
    return reduce(getitem, keys, tree)
