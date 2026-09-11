"""兼容旧命令名的 END 训练入口。

请使用 ``python train_denoise.py --config training.yaml``。实际实现位于
``train.py``，并且固定只训练论文原始卷积版 ``models/END.py``。
"""

from train import main


if __name__ == "__main__":
    main()
