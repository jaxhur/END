"""END 单卡复现实验的日志、checkpoint、颜色转换与复杂度工具。"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
import random
import sys
from time import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as functional


class BeijingFormatter(logging.Formatter):
    """把日志时间稳定转换为北京时间，而不是依赖服务器的系统时区。"""

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        from zoneinfo import ZoneInfo

        timestamp = datetime.fromtimestamp(record.created, tz=timezone.utc).astimezone(ZoneInfo("Asia/Shanghai"))
        return timestamp.strftime(datefmt or "%Y-%m-%d %H:%M:%S")


def create_logger(name: str, log_path: Path) -> logging.Logger:
    """创建终端和文件逐字符一致的 UTF-8 logger。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"{name}:{log_path.resolve()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = BeijingFormatter("%(asctime)s %(levelname)s: %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, mode="a", encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def seed_everything(seed: int) -> None:
    """为单卡实验设置 Python、NumPy 与 PyTorch 随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    """为 DataLoader worker 派生确定性的 Python/NumPy 随机种子。"""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    """将 YAML 中的相对路径解析为相对于项目根目录的绝对路径。"""
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def format_duration(seconds: float) -> str:
    """将秒数格式化为实验日志要求的 ``HH:MM:SS``。"""
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def pad_to_multiple(image: torch.Tensor, multiple: int = 4) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """将 NCHW 图像右侧和底侧 padding 到指定倍数，并返回原始高宽。"""
    if image.ndim != 4:
        raise ValueError(f"期望 NCHW Tensor，实际为 {tuple(image.shape)}")
    height, width = image.shape[-2:]
    pad_height = (multiple - height % multiple) % multiple
    pad_width = (multiple - width % multiple) % multiple
    if pad_height == 0 and pad_width == 0:
        return image, (height, width)
    # 极小图像无法使用 reflect padding，退回 replicate 保持推理可用。
    mode = "reflect" if height > 1 and width > 1 else "replicate"
    return functional.pad(image, (0, pad_width, 0, pad_height), mode=mode), (height, width)


def ycrcb_tensor_to_rgb_255(image: torch.Tensor) -> np.ndarray:
    """将 END 输出的 YCrCb ``[0,255]`` Tensor 转成 RGB ``[0,255]`` 浮点图像。"""
    if image.ndim == 4:
        if image.shape[0] != 1:
            raise ValueError("仅支持单张图像的颜色转换。")
        image = image[0]
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"期望 CHW 三通道 Tensor，实际为 {tuple(image.shape)}")
    ycrcb_01 = np.clip(image.detach().float().cpu().numpy().transpose(1, 2, 0), 0.0, 255.0) / 255.0
    bgr_01 = cv2.cvtColor(ycrcb_01, cv2.COLOR_YCrCb2BGR)
    return np.clip(bgr_01[..., ::-1] * 255.0, 0.0, 255.0)


def save_rgb_image(path: Path, image_rgb_255: np.ndarray) -> None:
    """按输入的规范化相对路径保存 RGB 图像，并自动创建父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = np.rint(np.clip(image_rgb_255[..., ::-1], 0.0, 255.0)).astype(np.uint8)
    if not cv2.imwrite(str(path), image_bgr):
        raise RuntimeError(f"无法保存增强图：{path}")


def _torch_load(path: Path, map_location: Any) -> Any:
    """兼容不同 PyTorch 版本读取 checkpoint。"""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def unwrap_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
    """从纯权重或旧版封装 checkpoint 中提取生成网络 state_dict。"""
    state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state_dict, dict):
        raise TypeError("checkpoint 不包含可用的 state_dict。")
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def load_generator_weights(model: torch.nn.Module, weights: Path, device: torch.device) -> None:
    """加载仅供推理/测试使用的 ``*_G.pth`` 生成网络权重。"""
    if not weights.is_file():
        raise FileNotFoundError(f"权重不存在：{weights}")
    model.load_state_dict(unwrap_state_dict(_torch_load(weights, device)), strict=True)


def save_generator_weights(model: torch.nn.Module, path: Path) -> None:
    """保存不含 optimizer 等训练状态的纯生成网络权重。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def find_latest_training_state(state_dir: Path) -> Optional[Path]:
    """按 checkpoint 内 global iteration 寻找最新训练状态。"""
    candidates = []
    for path in state_dir.glob("*.state"):
        try:
            state = _torch_load(path, "cpu")
            candidates.append((int(state.get("global_iter", -1)), path))
        except Exception:
            continue
    return max(candidates, default=(-1, None), key=lambda item: item[0])[1]


def save_training_state(path: Path, state: Dict[str, Any]) -> None:
    """保存可自动续训所需的模型进度、优化器、scheduler 与 AMP 状态。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_training_state(path: Path, device: torch.device) -> Dict[str, Any]:
    """读取一个 training state，并验证其基本类型。"""
    state = _torch_load(path, device)
    if not isinstance(state, dict):
        raise TypeError(f"训练状态格式错误：{path}")
    return state


def calculate_model_complexity(model: torch.nn.Module, device: torch.device) -> Dict[str, float | str]:
    """按固定 THOP、1x3x256x256 口径统计参数量、GMACs 与 GFLOPs。"""
    try:
        from thop import profile
    except ImportError as exc:
        raise RuntimeError("缺少 thop；请在服务器执行 pip install thop 后再测试。") from exc

    params_m = sum(parameter.numel() for parameter in model.parameters()) / 1e6
    was_training = model.training
    model.eval()
    dummy = torch.randn(1, 3, 256, 256, device=device)
    with torch.no_grad():
        macs, _ = profile(model, inputs=(dummy,), verbose=False)
    if was_training:
        model.train()
    return {
        "params_m": float(params_m),
        "gmacs_g": float(macs / 1e9),
        "gflops_g": float(2.0 * macs / 1e9),
        "input_size": "1x3x256x256",
        "complexity_tool": "thop.profile",
        "complexity_note": "GMACs=THOP返回值/1e9；GFLOPs=2*MACs/1e9",
    }


def elapsed_since(start_time: float, previous_elapsed: float = 0.0) -> float:
    """计算包含断点前累计时间的本次实验已训练时长。"""
    return previous_elapsed + (time() - start_time)
