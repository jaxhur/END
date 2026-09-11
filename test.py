"""兼容旧命令名的 END 配对测试入口。

请使用 ``python test.py --config training.yaml --weights <*_G.pth>``。测试
固定使用 ``models/END.py``，不再加载论文增强版 END+ 的 ``END_v2.py``。
"""

from test_end import main


if __name__ == "__main__":
    main()
