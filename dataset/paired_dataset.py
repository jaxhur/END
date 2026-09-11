"""成对低照度数据集及 END 所需的 YCrCb 数据转换。"""

from __future__ import annotations

from pathlib import Path
import random
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def _iter_image_files(root: Path) -> Iterable[Path]:
    """递归枚举图像文件，并按规范化相对路径保持稳定顺序。"""
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def _build_image_map(root: Path) -> Dict[str, Path]:
    """以相对于指定根目录的 POSIX 路径建立图像索引。"""
    if not root.is_dir():
        raise FileNotFoundError(f"图像目录不存在：{root}")

    mapping: Dict[str, Path] = {}
    for path in _iter_image_files(root):
        key = path.relative_to(root).as_posix()
        if key in mapping:
            raise RuntimeError(f"检测到重复的规范化相对路径：{key}")
        mapping[key] = path
    if not mapping:
        raise RuntimeError(f"图像目录为空或不含支持的图像文件：{root}")
    return mapping


def build_paired_paths(lq_root: str | Path, gt_root: str | Path) -> List[Tuple[str, Path, Path]]:
    """按规范化相对路径严格配对 LQ 与 GT 图像。

    Args:
        lq_root: 低照度图像根目录。
        gt_root: 正常曝光图像根目录。

    Returns:
        每项为 ``(relative_path, lq_path, gt_path)`` 的稳定排序列表。

    Raises:
        RuntimeError: 两侧图像集合存在缺失或多余文件时抛出，避免静默错配。
    """
    lq_root_path = Path(lq_root)
    gt_root_path = Path(gt_root)
    lq_map = _build_image_map(lq_root_path)
    gt_map = _build_image_map(gt_root_path)
    lq_keys = set(lq_map)
    gt_keys = set(gt_map)
    missing_gt = sorted(lq_keys - gt_keys)
    missing_lq = sorted(gt_keys - lq_keys)
    if missing_gt or missing_lq:
        examples = []
        if missing_gt:
            examples.append(f"GT 缺失 {len(missing_gt)} 项，例如：{missing_gt[:3]}")
        if missing_lq:
            examples.append(f"LQ 缺失 {len(missing_lq)} 项，例如：{missing_lq[:3]}")
        raise RuntimeError("LQ/GT 配对失败；" + "；".join(examples))
    return [(key, lq_map[key], gt_map[key]) for key in sorted(lq_keys)]


def read_bgr_image(path: Path) -> np.ndarray:
    """读取一张 BGR uint8 图像，读取失败时带路径报错。"""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取图像：{path}")
    return image


def bgr_to_ycrcb_tensor(image: np.ndarray) -> torch.Tensor:
    """将 BGR uint8 图像转换为 END 使用的 YCrCb 浮点张量。

    原项目在 ``torchvision.to_tensor`` 前将图像转为 ``float32``，因此网络
    实际接收的是 ``[0, 255]`` 而非 ``[0, 1]`` 的 YCrCb 数值；这里保留该
    数值语义，避免无意改变论文模型的训练尺度。
    """
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    return torch.from_numpy(ycrcb.transpose(2, 0, 1)).contiguous()


class PairedLowLightDataset(Dataset):
    """支持 LOL-v1 与 LOL-v2 目录的严格成对低照度数据集。

    训练模式使用相同坐标裁剪和几何增强，以保持 LQ/GT 的像素对齐；验证和
    测试模式保留完整原图，推理阶段再由调用方做可逆 padding。
    """

    def __init__(
        self,
        lq_root: str | Path,
        gt_root: str | Path,
        patch_size: int | None = None,
        training: bool = False,
    ) -> None:
        self.pairs = build_paired_paths(lq_root, gt_root)
        self.patch_size = patch_size
        self.training = training
        if self.training and (not isinstance(patch_size, int) or patch_size <= 0):
            raise ValueError("训练数据集必须提供正整数 patch_size。")

    def __len__(self) -> int:
        """返回严格配对后的样本数量。"""
        return len(self.pairs)

    def _paired_augment(self, lq: torch.Tensor, gt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """对配对图像应用同一随机裁剪、翻转和旋转。"""
        assert self.patch_size is not None
        _, height, width = lq.shape
        if gt.shape != lq.shape:
            raise RuntimeError(f"LQ/GT 尺寸不一致：{tuple(lq.shape)} 与 {tuple(gt.shape)}")
        if height < self.patch_size or width < self.patch_size:
            raise RuntimeError(
                f"图像尺寸 {height}x{width} 小于训练 patch {self.patch_size}x{self.patch_size}。"
            )

        top = random.randint(0, height - self.patch_size)
        left = random.randint(0, width - self.patch_size)
        lq = lq[:, top : top + self.patch_size, left : left + self.patch_size]
        gt = gt[:, top : top + self.patch_size, left : left + self.patch_size]

        augmentation = random.randint(0, 7)
        if augmentation & 1:
            lq, gt = lq.flip(1), gt.flip(1)
        if augmentation & 2:
            lq, gt = lq.flip(2), gt.flip(2)
        rotations = augmentation >> 2
        if rotations:
            lq, gt = torch.rot90(lq, rotations, (1, 2)), torch.rot90(gt, rotations, (1, 2))
        return lq, gt

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        """读取一组 LQ/GT 图像并返回其规范化相对路径。"""
        relative_path, lq_path, gt_path = self.pairs[index]
        lq = bgr_to_ycrcb_tensor(read_bgr_image(lq_path))
        gt = bgr_to_ycrcb_tensor(read_bgr_image(gt_path))
        if self.training:
            lq, gt = self._paired_augment(lq, gt)
        elif lq.shape != gt.shape:
            raise RuntimeError(f"LQ/GT 尺寸不一致：{lq_path} 与 {gt_path}")
        return {"lq": lq, "gt": gt, "relative_path": relative_path}
