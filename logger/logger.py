import logging
import logging.config
from pathlib import Path
from utils import read_json


def setup_logging(save_dir, log_config='logger/logger_config.json', default_level=logging.INFO):
    """
    ログ設定を初期化する関数
    """
    log_config = Path(log_config) # ログ設定ファイルのパスをPathオブジェクトに変換
    if log_config.is_file(): # 設定ファイルが実際に存在するかをチェック
        config = read_json(log_config) # 設定ファイルを読み込む
        # 設定ファイル内のハンドラー設定をループし，ログファイルの保存パスを更新
        for _, handler in config['handlers'].items():
            if 'filename' in handler: # ハンドラーにファイル名が指定されているかをチェック
                handler['filename'] = str(save_dir / handler['filename']) # ファイルの保存先の指定のディレクトリに変更

        logging.config.dictConfig(config) # 更新した設定を用いてロギングを設定
    else:
        # 設定ファイルが見つからない場合は警告を出力し，デフォルトのログレベルで基本設定を使用
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=default_level)
