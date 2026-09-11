"""仅使用 ``models/END.py`` 测试 END 的三 LOL 数据集统一指标。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict

import lpips
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from dataset.paired_dataset import PairedLowLightDataset
from metrics import calculate_psnr, calculate_ssim
from models.END import Illum_YCRCB_Denoise_IN
from utils.experiment_utils import (
    calculate_model_complexity,
    create_logger,
    load_generator_weights,
    pad_to_multiple,
    resolve_project_path,
    save_rgb_image,
    ycrcb_tensor_to_rgb_255,
)


def parse_args() -> argparse.Namespace:
    """解析配置和必须显式传入的 END 生成网络权重。"""
    parser = argparse.ArgumentParser(description="测试论文原始卷积版 END 模型")
    parser.add_argument("--config", required=True, type=Path, help="与训练一致的数据集 YAML")
    parser.add_argument("--weights", required=True, type=Path, help="明确指定的 *_G.pth 生成网络权重")
    return parser.parse_args()


def read_config(config_path: Path) -> Dict[str, Any]:
    """读取测试所需的实验、数据集与 DataLoader 配置。"""
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    # 测试 DataLoader 复用 training.workers，因此三个顶层配置段都不可缺失。
    if not isinstance(config, dict) or not {"experiment", "dataset", "training"}.issubset(config):
        raise ValueError(f"配置必须包含 experiment、dataset、training：{config_path}")
    return config


def write_metric_csv(path: Path, result: Dict[str, Any]) -> None:
    """以单行 CSV 固化完整测试集质量与复杂度指标。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(result))
        writer.writeheader()
        writer.writerow(result)


def main() -> None:
    """加载 END 权重，保存增强图，输出并记录固定统一评价指标。"""
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    weights_path = args.weights if args.weights.is_absolute() else project_root / args.weights
    if not weights_path.name.endswith("_G.pth"):
        raise ValueError("测试权重必须是生成网络 *_G.pth，不能传 training_state/*.state。")
    if not torch.cuda.is_available():
        raise RuntimeError("END 测试需要 CUDA GPU；请在 4090/5090 服务器上运行。")

    config = read_config(config_path)
    experiment, dataset_config = config["experiment"], config["dataset"]
    experiment_name, dataset_name = str(experiment["name"]), str(dataset_config["name"])
    result_dir = project_root / "test_result" / experiment_name / dataset_name
    enhanced_dir = result_dir / "enhanced"
    logger = create_logger(f"{experiment_name}_{dataset_name}", result_dir / "test.log")
    device = torch.device("cuda")
    logger.info(
        "[%s][TEST][ENV] torch=%s cuda_runtime=%s device=%s gpu=%s",
        experiment_name, torch.__version__, torch.version.cuda, device, torch.cuda.get_device_name(device),
    )

    dataset = PairedLowLightDataset(
        resolve_project_path(project_root, dataset_config["val_lq"]),
        resolve_project_path(project_root, dataset_config["val_gt"]),
        training=False,
    )
    logger.info(
        "[%s][TEST][CONFIG] data=%s/%s lq=%s gt=%s weights=%s",
        experiment_name,
        dataset_config["name"],
        dataset_config["val_split"],
        resolve_project_path(project_root, dataset_config["val_lq"]).as_posix(),
        resolve_project_path(project_root, dataset_config["val_gt"]).as_posix(),
        weights_path.resolve().as_posix(),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=int(config["training"].get("workers", 4)), pin_memory=True)
    model = Illum_YCRCB_Denoise_IN().to(device)
    load_generator_weights(model, weights_path, device)
    model.eval()
    complexity = calculate_model_complexity(model, device)
    lpips_model = lpips.LPIPS(net="alex", version="0.1").to(device).eval()
    psnr_scores, ssim_scores, lpips_scores = [], [], []

    with torch.no_grad():
        for batch in loader:
            lq = batch["lq"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)
            relative_path = Path(batch["relative_path"][0])
            padded_lq, (height, width) = pad_to_multiple(lq, multiple=4)
            _, _, restored = model(padded_lq)
            restored, gt = restored[..., :height, :width], gt[..., :height, :width]
            pred_rgb_255 = np.rint(ycrcb_tensor_to_rgb_255(restored))
            gt_rgb_255 = np.rint(ycrcb_tensor_to_rgb_255(gt))
            psnr_scores.append(calculate_psnr(pred_rgb_255, gt_rgb_255))
            ssim_scores.append(calculate_ssim(pred_rgb_255, gt_rgb_255))
            pred_lpips = torch.from_numpy(pred_rgb_255.transpose(2, 0, 1)).unsqueeze(0).to(device).float() / 127.5 - 1.0
            gt_lpips = torch.from_numpy(gt_rgb_255.transpose(2, 0, 1)).unsqueeze(0).to(device).float() / 127.5 - 1.0
            lpips_scores.append(float(lpips_model(pred_lpips, gt_lpips).item()))
            save_rgb_image(enhanced_dir / relative_path, pred_rgb_255)

    result = {
        "experiment": experiment_name,
        "dataset": dataset_name,
        "train_split": dataset_config["train_split"],
        "test_split": dataset_config["val_split"],
        "psnr": f"{float(np.mean(psnr_scores)):.4f}",
        "psnr_mode": "BasicSR-RGB-crop0",
        "ssim": f"{float(np.mean(ssim_scores)):.4f}",
        "ssim_mode": "BasicSR-RGB-channel-mean-crop0",
        "lpips": f"{float(np.mean(lpips_scores)):.4f}",
        "lpips_backbone": "alex",
        "lpips_version": "0.1",
        "lpips_range": "[-1,1]",
        "params_m": f"{complexity['params_m']:.4f}",
        "gmacs_g": f"{complexity['gmacs_g']:.4f}",
        "gflops_g": f"{complexity['gflops_g']:.4f}",
        "input_size": complexity["input_size"],
        "checkpoint": weights_path.resolve().as_posix(),
        "enhanced_images": enhanced_dir.relative_to(project_root).as_posix(),
        "metric_source": "test_end.py",
        "complexity_tool": complexity["complexity_tool"],
        "complexity_note": complexity["complexity_note"],
        "resize": "false",
        "gt_mean": "false",
        "self_ensemble": "false",
    }
    write_metric_csv(result_dir / "metric.csv", result)
    logger.info(
        "[%s][TEST] data=%s images=%d psnr=%s rgb_ssim=%s lpips_alex_v0.1=%s Params(M)=%s GMACs(G)=%s GFLOPs(G)=%s",
        experiment_name, dataset_name, len(dataset), result["psnr"], result["ssim"], result["lpips"],
        result["params_m"], result["gmacs_g"], result["gflops_g"],
    )


if __name__ == "__main__":
    main()
