import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict


def ensure_dir(dirname):
    """
    指定されたディレクトリが存在しなければ作成する
    """
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False) # ディレクトリが存在しない場合、親ディレクトリも含めて作成

def read_json(fname):
    """
    JSONファイルを読み込み、順序付き辞書として返す
    """
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict) # ファイルを開いてJSONデータを順序付き辞書として読み込む

def write_json(content, fname):
    """
    辞書をJSONファイルとして書き込む
    """
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False) # JSONデータを整形してファイルに書き込む

def inf_loop(data_loader):
    """
    データローダーを無限に繰り返すラッパー関数
    """
    for loader in repeat(data_loader):
        yield from loader # データローダーからデータを無限に生成

def prepare_device(n_gpu_use):
    """
    利用可能なGPUを設定し、使用するGPUのインデックスリストを返す
    """
    n_gpu = torch.cuda.device_count() # 利用可能なGPUの数を取得
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu') # 使用するデバイスを設定
    list_ids = list(range(n_gpu_use)) # 使用するGPUのインデックスリスト
    return device, list_ids

class MetricTracker:
    """
    メトリクスの追跡を管理するクラス
    """
    def __init__(self, *keys, writer=None):
        self.writer = writer # TensorBoardライター（オプショナル）
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        """
        メトリクスデータをリセットする
        """
        for col in self._data.columns:
            self._data[col].values[:] = 0 # 各列を0にリセット

    def update(self, key, value, n=1):
        """
        特定のキーのメトリクスを更新する
        """
        if self.writer is not None:
            self.writer.add_scalar(key, value) # TensorBoardに値を書き込む
        self._data.total[key] += value * n # 合計値を更新
        self._data.counts[key] += n # カウントを更新
        self._data.average[key] = self._data.total[key] / self._data.counts[key] # 平均を計算

    def avg(self, key):
        """
        指定したキーの平均値を取得
        """
        return self._data.average[key]

    def result(self):
        """
        全メトリクスの平均値を辞書形式で返す
        """
        return dict(self._data.average)
