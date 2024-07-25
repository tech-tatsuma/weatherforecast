import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1

def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = 3
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        self.predict_linear = nn.Linear(
            self.seq_len, self.pred_len + self.seq_len)
        self.projection = nn.Linear(
            configs.d_model, configs.c_out, bias=True)

        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(
              configs.d_model * configs.seq_len, configs.num_class)

    def classification(self, x_enc, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]

class Configs:
    def __init__(self):
        self.seq_len = 3  # 入力シーケンスの長さ（3日分のデータ）
        self.label_len = 1
        self.pred_len = 1  # 予測する期間の長さ（次の1日）
        self.d_model = 64  # モデルの内部次元
        self.d_ff = 256  # Feed Forwardネットワークの次元
        self.num_kernels = [10, 20, 40]  # Inception block内の異なるカーネルサイズ
        self.e_layers = 2  # モデルにおけるエンコーダ/デコーダ層の数
        self.enc_in = 4  # 入力特徴量の数（meantemp, humidity, wind_speed, meanpressure）
        self.c_out = 1  # 出力特徴量の数（次の日のmeantemp）
        self.embed = 'timeF'  # 埋め込みタイプ（時間周波数埋め込み）
        self.freq = 'h'  # 周波数の単位（時）
        self.dropout = 0.1  # ドロップアウト率

config = Configs()
model = Model(config)

class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.model = model

    def forward(self, x):
        # xの形状: [batch_size, num_sequences, sequence_length, num_features]
        batch_size, num_sequences, seq_len, num_features = x.shape
        
        # 各時系列データに対して特徴抽出
        extracted_features = []
        for i in range(num_sequences):
            # 時系列データをmodelの入力に合わせる
            seq_input = x[:, i, :, :]  # [batch_size, sequence_length, num_features]
            seq_input = seq_input.unsqueeze(1)  # ダミーの"channel"次元を追加

            # forecast関数を使用して特徴を抽出
            features = self.model.forecast(seq_input, None, None, None)  # config設定に基づき適宜調整が必要
            extracted_features.append(features)

        # 特徴量のリストを結合
        features_concat = torch.cat(extracted_features, dim=-1)  # 最後の次元に沿って結合

        # 最終的な出力を生成するための全結合層
        final_output = self.model.projection(features_concat.squeeze(1))  # 不要な次元を削除

        return final_output

feature_extractor = FeatureExtractor(model)