# WeatherForecast
本レポジトリは，過去30日間の平均気温，湿度，風速，気圧のデータから次の日の天気を予測する機械学習モデルを構築するものです．
## Usage
### train
以下のコマンドを実行し，学習を実行してください．
```bash
python train.py -c configs/train.json
```
### パラメータチューニング
ベイズ最適化を用いたパラメータチューニングは以下のコマンドを実行し，学習してください．
```bash
python paramtune.py -c configs/paramtune.json
```
パラメータチューニングの際は`base/base_trainer.py`の中の
```bash
torch.save(state, str(self.checkpoint_dir / 'best.pth'))
```
をコメントアウトし，
```bash
torch.save(state, "best.pth")
```
のコメントアウトを消すようにしてください．