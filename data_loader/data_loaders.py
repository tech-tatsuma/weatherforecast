import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from base import BaseDataLoader
import torch

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

        # 異常値の処理
        self.bounds = {}
        self.handle_outliers(self.data)

        # データの正規化パラメータを取得（ただし、meantempのy用は除外）
        self.min_max_values = {}
        self.calculate_min_max_values(self.data)

        # 特徴量の正規化
        self.normalize_features(self.data)
        
        # データの準備
        for i in range(len(self.data) - 30):  # 最後から8日分は使用できないため除外
            feature_data = []
            for feature in self.features:
                feature_data.append(self.data.loc[i:i+29, feature].values)
            self.X.append(feature_data)
            self.y.append(self.data.loc[i+30, 'meantemp'])  # 30日目のmeantempをラベルとして追加

        print("Bounds for Outliers:", self.bounds)
        print("Min-Max Values for Normalization:", self.min_max_values)

    def handle_outliers(self, data):
        for column in self.features:
            q1 = data[column].quantile(0.25)
            q3 = data[column].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            self.bounds[column] = (lower_bound, upper_bound)
            data.loc[data[column] < lower_bound, column] = data[column].mean()
            data.loc[data[column] > upper_bound, column] = data[column].mean()

    def calculate_min_max_values(self, data):
        for column in self.features:
            if column != 'meantemp':  # meantempのy用正規化を避ける
                min_val = data[column].min()
                max_val = data[column].max()
                self.min_max_values[column] = (min_val, max_val)

    def normalize_features(self, data):
        for column in self.features:
            if column in self.min_max_values:  # meantempのy用は除外
                min_val, max_val = self.min_max_values[column]
                data[column] = (data[column] - min_val) / (max_val - min_val)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 特徴量とターゲットを返す
        features = torch.tensor(np.array(self.X[idx]), dtype=torch.float32)
        target = torch.tensor(np.array([self.y[idx]]), dtype=torch.float32)
        return features, target

class CustomDataLoader(BaseDataLoader):
    def __init__(self, filepath, batch_size=64, shuffle=True, validation_split=0.2, num_workers=0):
        dataset = WeatherDataset(filepath)
        super().__init__(dataset, batch_size, shuffle, validation_split, num_workers)
