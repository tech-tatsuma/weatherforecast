import torch.nn as nn
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    すべてのモデルの基底クラス
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        フォワードパスのロジック

        :return: モデルの出力
        """
        raise NotImplementedError # このクラスのメソッドはサブクラスで実装が必要

    def __str__(self):
        """
        トレーニング可能なパラメータの数を含むモデルの情報を文字列として出力
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters()) # トレーニング可能なパラメータのみを取得
        params = sum([np.prod(p.size()) for p in model_parameters]) # トレーニング可能なパラメータの総数を計算
        return super().__str__() + '\nTrainable parameters: {}'.format(params) # 基底クラスの文字列表現にパラメータ数を追加して返す
