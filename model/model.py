import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1

def FFT_for_Period(x, k=2):
    # [B, T, C]
    # xに対して時間軸に沿って実数高速フーリエ変換を実行
    xf = torch.fft.rfft(x, dim=1)

    # find period by amplitudes
    # xfの絶対値を取り，最初にバッチ次元，次にチャンネル次元に沿って平均を計算し，各周波成分の平均振幅を求める．
    frequency_list = abs(xf).mean(0).mean(-1)
    # 直流成分(0周波数成分)を0に設定し，無視する
    frequency_list[0] = 0
    # 振幅が最大の上位k個の周波数成分を選出する
    _, top_list = torch.topk(frequency_list, k)
    # GPU上で計算されている場合，top_listをCPUに移動させ，NumPy配列に変換する
    top_list = top_list.detach().cpu().numpy()
    # 選出された周波数成分に基づいて周期を計算する．時間軸の長さを最大周波数インデックスで割ることで周期を求める．
    period = x.shape[1] // top_list
    # 計算された周期と選択された周波数成分の平均振幅を返す．
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        # 入力シーケンスの長さ
        self.seq_len = configs.seq_len
        # 予測するシーケンスの長さ
        self.pred_len = configs.pred_len
        # 上位k個の周期を抽出
        self.k = configs.top_k
        # パラメータの効率的な設計
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        # バッチサイズ，シーケンス長，特徴料の次元を取得
        B, T, N = x.size()

        # FFT_for_Period関数を用いて周期を計算
        period_list, period_weight = FFT_for_Period(x, self.k)

        # 結果を格納するリスト
        res = []
        for i in range(self.k):
            # i番目の周期を取得
            period = period_list[i]
            # パディング処理
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)) + 1, x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x

            # reshapeと2d畳み込み
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous() # 128, 2, 9, 64
            # 畳み込み層を通す
            out = self.conv(out)
            # 元の形状に戻す
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            # 予測長だけを切り取りだして追加
            res.append(out[:, :(self.seq_len + self.pred_len - 1), :])
        # k個の結果を最後の次元に沿って積み重ねる
        res = torch.stack(res, dim=-1)
        # 適応的な集約
        # 周期の重みをソフトマックスで正規化
        period_weight = F.softmax(period_weight, dim=1)
        # 重みを拡張して各要素に適用
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        # 重み付き和を計算
        res = torch.sum(res * period_weight, -1)
        # 残差接続
        res = res + x
        return res


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        # 設定を保存
        self.configs = configs
        # 入力系列の長さ
        self.seq_len = configs.seq_len
        # ラベルの長さ
        self.label_len = configs.label_len
        # 予測する長さ
        self.pred_len = configs.pred_len

        # TimesBlockのリストを構築
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # TimesBlockの数
        self.layer = configs.e_layers
        # 層の正規化
        self.layer_norm = nn.LayerNorm(configs.d_model)

        # 予測層
        self.predict_linear = nn.Linear(
            self.seq_len, self.pred_len + self.seq_len)

        # 活性化関数GELU
        self.act = F.gelu
        # ドロップアウト
        self.dropout = nn.Dropout(configs.dropout)
        # 出力のための線形変換
        self.projection = nn.Linear(
              configs.d_model * configs.seq_len, configs.num_class)

    def classification(self, x_enc, x_mark_enc):
        # 埋め込み層を通してデータを変換
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesBlockを通して特徴抽出
        for i in range(self.layer):
            # 各TimeBlockを適用後に層を正規化
            enc_out = self.layer_norm(self.model[i](enc_out))

        # 活性化関数を適用
        output = self.act(enc_out)
        # ドロップアウトを適用
        output = self.dropout(output)
        # パディング部分を0に設定
        # output = output * x_mark_enc.unsqueeze(-1)
        # バッチサイズと特徴量次元を統合
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        # 分類のための線形変換
        output = self.projection(output)  # (batch_size, num_classes)
        return output


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # 分類関数を通じてデコーダー出力を取得
        dec_out = self.classification(x_enc, x_mark_enc)
        return dec_out  # [B, N]

class Configs:
    def __init__(self):
        self.seq_len = 30  # 入力シーケンスの長さ（8日分のデータ）
        self.label_len = 1  # ラベル長さ
        self.pred_len = 1  # 予測する期間の長さ（次の1日）
        self.d_model = 64  # モデルの内部次元
        self.d_ff = 256  # Feed Forwardネットワークの次元
        self.num_kernels = 4  # Inception block内の異なるカーネルサイズ
        self.e_layers = 3  # モデルにおけるエンコーダ/デコーダ層の数
        self.enc_in = 4  # 入力特徴量の数（meantemp, humidity, wind_speed, meanpressure）
        self.c_out = 1  # 出力特徴量の数（次の日のmeantemp）
        self.embed = 'timeF'  # 埋め込みタイプ（時間周波数埋め込み）
        self.freq = 'h'  # 周波数の単位（時）
        self.dropout = 0  # ドロップアウト率
        self.num_class = 1  # 分類するクラスの数
        self.top_k = 2 # FFT分析で上位k個の周期を考慮

config = Configs()
model = Model(config)
    
class WetherForecastModel(nn.Module):
    def __init__(self, model):
        super(WetherForecastModel, self).__init__()
        # Modelインスタンスを取得
        self.model = model

    def forward(self, x, seq_x_mark):
        # データの次元を入れ替え
        x = x.permute(0, 2, 1)
        # モデルを通して特徴量を抽出
        output = self.model(x, seq_x_mark, None, None)
        
        return output

# モデルをインスタンス化
feature_extractor = WetherForecastModel(model)