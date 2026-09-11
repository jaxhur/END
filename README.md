# END：统一三 LOL 数据集复现实验

本目录只训练和测试论文的原始卷积版 **END**，统一使用`models/END.py`。

`models/END_v2.py` 是论文的增强版 END+，保留源码供论文对照，但不会被当前训练或配对测试入口加载。

# 创建环境

```bash
git clone https://github.com/jaxhur/END.git
# 仅首次执行：创建并激活独立环境。
conda create -n end python=3.10 -y
conda activate end


python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install 
python -m pip install -r requirements.txt
```



# 数据集

```bash
apt-get install -y unzip
pip install gdown
cd dataset
gdown "https://drive.google.com/uc?id=1mAN3ll5wWwt1Xz0C7uio31-NJu-50S8Z" -O LOL-v1.zip
gdown "https://drive.google.com/uc?id=1L0UnJg6gZ4Eb7It2EuNxP0L3lQNmKMaP" -O LOL-v2-renamed.zip

# 解压至当前训练配置实际读取的目录。
unzip -q LOL-v1.zip -d LOL-v1
unzip -q LOL-v2-renamed.zip -d LOL-v2
cd ..
```

最终数据布局必须为：

```text
dataset/LOL-v1/our485/{low,high}
dataset/LOL-v1/eval15/{low,high}

dataset/LOL-v2/Synthetic/Train/{Low,Normal}
dataset/LOL-v2/Synthetic/Test/{Low,Normal}

dataset/LOL-v2/Real_captured/Train/{Low,Normal}
dataset/LOL-v2/Real_captured/Test/{Low,Normal}
```

# 训练

| 数据集 | YAML | Train / Test split | BatchSize | PatchSize | Epochs |
|---|---|---|---:|---:|---:|
| LOL-v1 | `training.yaml` | `our485` / `eval15` | 16 | 128 | 5000 |
| LOL-v2-syn | `training_lolv2_syn.yaml` | `Synthetic/Train` / `Synthetic/Test` | 16 | 128 | 5000 |
| LOL-v2-real | `training_lolv2_real.yaml` | `Real_captured/Train` / `Real_captured/Test` | 16 | 128 | 5000 |

## 训练

```bash
# 只选一张可见卡；不要在代码中修改 CUDA device index。
CUDA_VISIBLE_DEVICES=0 python train_denoise.py --config training.yaml
CUDA_VISIBLE_DEVICES=0 python train_denoise.py --config training_lolv2_syn.yaml
CUDA_VISIBLE_DEVICES=0 python train_denoise.py --config training_lolv2_real.yaml
```



```text
experiments/<实验名>/
  config.yaml
  models/latest_G.pth
  models/best_G.pth
  models/<iter>_G.pth
  training_state/<iter>.state
  logs/train.log
  logs/val.log
  tb_looger/
```



# 测试

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --config training.yaml --weights experiments/END_LOL-v1/models/best_G.pth

CUDA_VISIBLE_DEVICES=0 python test.py --config training_lolv2_syn.yaml --weights experiments/END_LOL-v2-syn/models/best_G.pth

CUDA_VISIBLE_DEVICES=0 python test.py --config training_lolv2_real.yaml --weights experiments/END_LOL-v2-real/models/best_G.pth
```

每次测试会保存全部增强图，并在以下目录写入日志和单行 `metric.csv`：

```text
test_result/<实验名>/<数据集名>/
  enhanced/<与 LQ 相同的规范化相对路径>
  metric.csv
  test.log
```

`metric.csv` 固定记录完整测试集逐图算术平均后的：

- PSNR：BasicSR 等价 RGB 联合 MSE，`[0,255]`、`crop_border=0`；
- SSIM：BasicSR 等价 RGB 三通道分别计算后平均，`11x11`、`sigma=1.5`；
- LPIPS：AlexNet、version `0.1`、RGB `[-1,1]`；
- Params(M)：全部 END 生成网络参数除以 `1e6`；
- GMACs(G) / GFLOPs(G)：THOP、`1x3x256x256`，其中 `GFLOPs=2*MACs/1e9`。



