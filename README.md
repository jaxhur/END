# END：统一三 LOL 数据集复现实验

本目录只训练和测试论文的原始卷积版 **END**，统一使用
`models/END.py`。`models/END_v2.py` 是论文的增强版 END+，保留源码供
论文对照，但不会被当前训练或配对测试入口加载。

## 数据目录

`dataset/` 同时是 Python 数据集包和本项目的数据根目录。代码不会创建、
移动或下载数据；请在服务器把数据放成下面的固定结构：

```text
dataset/LOL-v1/our485/{low,high}
dataset/LOL-v1/eval15/{low,high}

dataset/LOL-v2/Synthetic/Train/{Low,Normal}
dataset/LOL-v2/Synthetic/Test/{Low,Normal}

dataset/LOL-v2/Real_captured/Train/{Low,Normal}
dataset/LOL-v2/Real_captured/Test/{Low,Normal}
```

LQ 与 GT 通过相对于各自根目录的规范化路径严格配对。缺图、重复相对路径或
图像尺寸不一致会直接报错，不会按目录顺序静默错配。

## 环境：RTX 4090 / RTX 5090 单卡

项目没有自定义 CUDA 扩展，也没有写死 GPU 型号、显存或 device index。应在
每台目标服务器安装 **CUDA 12.8 或更高**对应的 PyTorch CUDA build，并在执行
前确认 `torch.version.cuda`、NVIDIA driver 与实际 GPU。

```bash
# 示例：先按 PyTorch 官方 CUDA 12.8 wheel 源安装 torch 与 torchvision。
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
pip install -r requirements.txt

# 记录服务器端实际运行时，而不要只看 nvcc。
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

首次调用 LPIPS-Alex-v0.1 时，服务器需要能读取 LPIPS/AlexNet 权重缓存；若
服务器离线，请预先准备相应缓存。完整训练前，分别在实际租用的 4090 或 5090
上完成单 batch 前向、反向与 AMP 冒烟验证。三份 YAML 默认 `amp: false`，以
保留论文原始数值路径；只有在目标卡验证后才建议改为 `true`。

## 三套论文训练参数

论文训练细节采用 `5000 epochs`、`batch size=16`、`128x128` 成对随机裁剪。
三份 YAML 均保持这组设置，实际总 iteration 在数据集加载后由
`ceil(训练图数 / 16) * 5000` 动态计算并写入 `train.log`。

| 数据集 | YAML | Train / Test split | BatchSize | PatchSize | Epochs |
|---|---|---|---:|---:|---:|
| LOL-v1 | `training.yaml` | `our485` / `eval15` | 16 | 128 | 5000 |
| LOL-v2-syn | `training_lolv2_syn.yaml` | `Synthetic/Train` / `Synthetic/Test` | 16 | 128 | 5000 |
| LOL-v2-real | `training_lolv2_real.yaml` | `Real_captured/Train` / `Real_captured/Test` | 16 | 128 | 5000 |

## 训练

训练只需指定 YAML。开始时会优先扫描
`experiments/<实验名>/training_state/*.state`，自动从最大的 global iteration
恢复模型、optimizer、scheduler、AMP scaler、最佳 PSNR 及对应 RGB SSIM。若 state
被清理但仍有 `models/latest_G.pth`，会自动加载该纯权重并在日志中明确标记为
“仅权重恢复”；此时 optimizer、scheduler 和 scaler 会重新初始化。

```bash
# 只选一张可见卡；不要在代码中修改 CUDA device index。
CUDA_VISIBLE_DEVICES=0 python train_denoise.py --config training.yaml
CUDA_VISIBLE_DEVICES=0 python train_denoise.py --config training_lolv2_syn.yaml
CUDA_VISIBLE_DEVICES=0 python train_denoise.py --config training_lolv2_real.yaml
```

每 20 个 global iteration 输出一行训练状态；约每 1000 iter 在对应完整测试集
验证 RGB PSNR/SSIM。训练中维护：

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

`*_G.pth` 是纯生成网络权重，只用于测试或推理；`.state` 包含训练恢复状态，
不可传给测试命令。`best_G.pth` 仅在完整验证集 PSNR 严格提升时更新。

## 测试与 metric.csv

测试权重必须显式传入，不会自动猜测 latest 或 best。推理期间仅对输入做可逆
padding，保存与计分前会裁回原尺寸；不会 resize、GT-Mean 或 self-ensemble。

```bash
CUDA_VISIBLE_DEVICES=0 python test.py --config training.yaml \
  --weights experiments/END_LOL-v1/models/best_G.pth

CUDA_VISIBLE_DEVICES=0 python test.py --config training_lolv2_syn.yaml \
  --weights experiments/END_LOL-v2-syn/models/best_G.pth

CUDA_VISIBLE_DEVICES=0 python test.py --config training_lolv2_real.yaml \
  --weights experiments/END_LOL-v2-real/models/best_G.pth
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

## Citation

```text
@ARTICLE{10718327,
  author={Wang, Huake and Yan, Xiaoyang and Hou, Xingsong and Zhang, Kaibing and Dun, Yujie},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  title={Extracting Noise and Darkness: Low-Light Image Enhancement via Dual Prior Guidance},
  year={2025},
  volume={35},
  number={2},
  pages={1700--1714},
  doi={10.1109/TCSVT.2024.3480930}}
```
