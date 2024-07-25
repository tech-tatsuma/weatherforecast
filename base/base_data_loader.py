import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    """
    すべてのデータローダーの基底クラス
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        # バリデーションセットの分割比率
        self.validation_split = validation_split
        # データをシャッフルするかどうかのフラグ
        self.shuffle = shuffle

        # バッチのインデックス
        self.batch_idx = 0
        # データセットのサンプル数
        self.n_samples = len(dataset)

        # サンプラーの分割
        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs) # DataLoaderの初期化

    def _split_sampler(self, split):

        if split == 0.0: # 分割が0の場合バリデーションは作成しない
            return None, None

        # 全データのインデックス
        idx_full = np.arange(self.n_samples)

        # 乱数シード固定
        np.random.seed(0)
        # インデックスをシャッフル
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split) # バリデーションセットのサイズを計算

        # バリデーションセットのインデックス
        valid_idx = idx_full[0:len_valid]
        # トレーニングセットのインデックス
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        # トレーニングセット用のサンプラー
        train_sampler = SubsetRandomSampler(train_idx)
        # バリデーションセット用のサンプラー
        valid_sampler = SubsetRandomSampler(valid_idx)

        # サンプラー使用時はシャッフルを無効にする
        self.shuffle = False
        # トレーニングサンプルの数を更新
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None # バリデーションサンプラーがない場合はNoneを返す
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs) # バリデーションデータローダーを生成して返す
