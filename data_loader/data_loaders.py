import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class WeatherDataset(Dataset):
    """日次デリー気象データのためのカスタムデータセットクラス
    """

    def __init__(self, filepath):
        # データの読み込み
        self.data = pd.read_csv(filepath)

        # 次の日のmeantempを予測するために必要なカラムを選択
        self.features = ['meantemp', 'humidity', 'wind_speed', 'meanpressure']

        # 特徴量とラベルを準備
        self.X = []
        self.y = []
        for i in range(len(self.data) - 3):  # 最後から3日分は使用できないため除外
            # 各特徴量ごとに3日分のデータを抽出し、リストとして追加（シーケンスデータとして処理）
            feature_data = []
            for feature in self.features:
                feature_data.append(self.data.loc[i:i+2, feature].values)
            self.X.append(feature_data)
            self.y.append(self.data.loc[i+3, 'meantemp'])  # 4日目のmeantempをラベルとして追加

    def __len__(self):
        # データセットの長さは最初と最後のデータを除いた長さ
        return len(self.X)

    def __getitem__(self, idx):
        # 特徴量とターゲットを返す
        features = np.array(self.X[idx], dtype=np.float32)
        target = np.array([self.y[idx]], dtype=np.float32)  # ラベルを配列に変換
    return features, target

class CustomDataLoader(BaseDataLoader):
    def __init__(self, filepath, batch_size=64, shuffle=True, validation_split=0.2, num_workers=0):
        dataset = WeatherDataset(filepath)
        super().__init__(dataset, batch_size, shuffle, validation_split, num_workers)