"""仅复现论文 END 模型的单卡三 LOL 数据集训练入口。"""

from __future__ import annotations

import argparse
from datetime import datetime
import math
from pathlib import Path
import shutil
from time import time
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter
import yaml

import losses
from dataset.paired_dataset import PairedLowLightDataset
from metrics import calculate_psnr, calculate_ssim
from models.END import Illum_YCRCB_Denoise_IN
from utils.experiment_utils import (
    calculate_model_complexity,
    create_logger,
    elapsed_since,
    find_latest_training_state,
    format_duration,
    load_generator_weights,
    load_training_state,
    pad_to_multiple,
    resolve_project_path,
    save_generator_weights,
    save_training_state,
    seed_everything,
    seed_worker,
    ycrcb_tensor_to_rgb_255,
)


def parse_args() -> argparse.Namespace:
    """解析训练 YAML；GPU 由外部可见设备和 PyTorch 决定。"""
    parser = argparse.ArgumentParser(description="训练论文原始卷积版 END 模型")
    parser.add_argument("--config", required=True, type=Path, help="训练 YAML 配置文件")
    return parser.parse_args()


def read_config(config_path: Path) -> Dict[str, Any]:
    """读取并校验训练配置。"""
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict) or not {"experiment", "dataset", "training"}.issubset(config):
        raise ValueError(f"配置必须包含 experiment、dataset、training：{config_path}")
    return config


class EpochOffsetSampler(Sampler[int]):
    """为每个 epoch 生成确定性乱序索引，并从 checkpoint 偏移继续。"""

    def __init__(self, dataset_size: int, seed: int, epoch: int, start_offset: int = 0) -> None:
        self.dataset_size = dataset_size
        self.seed = seed
        self.epoch = epoch
        self.start_offset = start_offset

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.dataset_size, generator=generator).tolist()
        return iter(indices[self.start_offset :])

    def __len__(self) -> int:
        return max(0, self.dataset_size - self.start_offset)


def create_train_loader(
    dataset: PairedLowLightDataset,
    batch_size: int,
    workers: int,
    seed: int,
    epoch: int,
    start_step: int,
) -> DataLoader:
    """构造从指定 epoch/batch 偏移恢复的确定性训练 DataLoader。"""
    generator = torch.Generator()
    generator.manual_seed(seed + epoch)
    sampler = EpochOffsetSampler(len(dataset), seed, epoch, start_step * batch_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def create_scheduler(optimizer: torch.optim.Optimizer, total_iters: int, warmup_iters: int, min_lr: float) -> LambdaLR:
    """创建 warmup 后余弦退火的 iteration 调度器。"""
    initial_lr = optimizer.param_groups[0]["lr"]
    min_ratio = min_lr / initial_lr

    def lr_lambda(step: int) -> float:
        if warmup_iters and step < warmup_iters:
            return float(step + 1) / warmup_iters
        progress = min(max((step - warmup_iters) / max(1, total_iters - warmup_iters), 0.0), 1.0)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def run_validation(model: nn.Module, val_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """在完整验证集上按固定 RGB 口径计算逐图平均 PSNR、SSIM。"""
    model.eval()
    psnr_scores, ssim_scores = [], []
    with torch.no_grad():
        for batch in val_loader:
            lq = batch["lq"].to(device, non_blocking=True)
            gt = batch["gt"].to(device, non_blocking=True)
            padded_lq, (height, width) = pad_to_multiple(lq, multiple=4)
            _, _, restored = model(padded_lq)
            restored = restored[..., :height, :width]
            gt = gt[..., :height, :width]
            pred_rgb = np.rint(ycrcb_tensor_to_rgb_255(restored))
            gt_rgb = np.rint(ycrcb_tensor_to_rgb_255(gt))
            psnr_scores.append(calculate_psnr(pred_rgb, gt_rgb))
            ssim_scores.append(calculate_ssim(pred_rgb, gt_rgb))
    model.train()
    if not psnr_scores:
        raise RuntimeError("验证集为空，无法计算完整测试集指标。")
    return float(np.mean(psnr_scores)), float(np.mean(ssim_scores))


def make_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler,
    epoch: int,
    step: int,
    global_iter: int,
    elapsed_seconds: float,
    best_psnr: float,
    best_rgb_ssim: float | None,
) -> Dict[str, Any]:
    """打包自动续训所需的模型、优化器、调度器、AMP 和最佳指标。"""
    return {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "step": step,
        "global_iter": global_iter,
        "elapsed_seconds": elapsed_seconds,
        "best_psnr": best_psnr,
        "best_rgb_ssim": best_rgb_ssim,
    }


def main() -> None:
    """执行 END 单卡训练、周期验证和自动断点续训。"""
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    config = read_config(config_path)
    experiment, dataset_config, training = config["experiment"], config["dataset"], config["training"]
    if not torch.cuda.is_available():
        raise RuntimeError("END 训练需要 CUDA GPU；请在 4090/5090 服务器上运行。")

    device = torch.device("cuda")
    experiment_name = str(experiment["name"])
    experiment_dir = project_root / "experiments" / experiment_name
    models_dir, state_dir, logs_dir = experiment_dir / "models", experiment_dir / "training_state", experiment_dir / "logs"
    for directory in (models_dir, state_dir, logs_dir, experiment_dir / "tb_looger"):
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, experiment_dir / "config.yaml")
    train_logger = create_logger(experiment_name, logs_dir / "train.log")
    val_logger = create_logger(experiment_name, logs_dir / "val.log")
    server_now = datetime.now().astimezone()
    train_logger.info(
        "[%s][ENV] server_tz=%s utc_offset=%s",
        experiment_name,
        server_now.tzname(),
        server_now.strftime("%z"),
    )
    train_logger.info(
        "[%s][ENV] torch=%s cuda_runtime=%s device=%s gpu=%s",
        experiment_name, torch.__version__, torch.version.cuda, device, torch.cuda.get_device_name(device),
    )

    seed = int(experiment.get("seed", 1234))
    seed_everything(seed)
    torch.backends.cudnn.benchmark = bool(training.get("cudnn_benchmark", True))
    train_dataset = PairedLowLightDataset(
        resolve_project_path(project_root, dataset_config["train_lq"]),
        resolve_project_path(project_root, dataset_config["train_gt"]),
        patch_size=int(training["patch_size"]), training=True,
    )
    val_dataset = PairedLowLightDataset(
        resolve_project_path(project_root, dataset_config["val_lq"]),
        resolve_project_path(project_root, dataset_config["val_gt"]), training=False,
    )
    batch_size, workers = int(training["batch_size"]), int(training.get("workers", 4))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=workers, pin_memory=True)
    steps_per_epoch, total_epochs = math.ceil(len(train_dataset) / batch_size), int(training["epochs"])
    total_iters = steps_per_epoch * total_epochs

    model = Illum_YCRCB_Denoise_IN().to(device)
    optimizer = Adam(model.parameters(), lr=float(training["lr_initial"]), betas=(0.9, 0.999), eps=1e-8)
    scheduler = create_scheduler(
        optimizer, total_iters, int(training.get("warmup_epochs", 3)) * steps_per_epoch, float(training["lr_min"])
    )
    amp_enabled = bool(training.get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    char_criterion, ssim_criterion = losses.CharbonnierLoss(), losses.SSIM_Loss()
    writer = SummaryWriter(log_dir=str(experiment_dir / "tb_looger"))

    start_epoch, resume_step, global_iter, previous_elapsed = 1, 0, 0, 0.0
    best_psnr, best_rgb_ssim = float("-inf"), None
    latest_state = find_latest_training_state(state_dir)
    if latest_state:
        state = load_training_state(latest_state, device)
        model.load_state_dict(state["state_dict"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state.get("scaler", {}))
        start_epoch, resume_step, global_iter = int(state["epoch"]), int(state.get("step", 0)), int(state["global_iter"])
        if resume_step >= steps_per_epoch:
            start_epoch, resume_step = start_epoch + 1, 0
        previous_elapsed = float(state.get("elapsed_seconds", 0.0))
        best_psnr, best_rgb_ssim = float(state.get("best_psnr", float("-inf"))), state.get("best_rgb_ssim")
        train_logger.info(
            "[%s][RESUME] state=%s epoch=%d skip_step=%d iter=%d",
            experiment_name, latest_state.name, start_epoch, resume_step, global_iter,
        )
    else:
        # 若状态文件被清理但 latest 权重仍在，至少保留网络参数；优化器和调度器
        # 无法从纯权重恢复，故从新的训练进度重新开始并明确写入日志。
        latest_weights = models_dir / "latest_G.pth"
        if latest_weights.is_file():
            load_generator_weights(model, latest_weights, device)
            train_logger.info(
                "[%s][RESUME] weights=%s only; optimizer/scheduler/scaler reset, epoch=1 iter=0",
                experiment_name,
                latest_weights.name,
            )

    complexity = calculate_model_complexity(model, device)
    train_logger.info(
        "[%s][MODEL] Params(M)=%.4f GMACs(G)=%.4f GFLOPs(G)=%.4f input=%s",
        experiment_name, complexity["params_m"], complexity["gmacs_g"], complexity["gflops_g"], complexity["input_size"],
    )
    train_logger.info(
        "[%s][CONFIG] data=%s train=%d val=%d batch=%d patch=%dx%d epochs=%d steps_per_epoch=%d total_iters=%d",
        experiment_name, dataset_config["name"], len(train_dataset), len(val_dataset), batch_size,
        int(training["patch_size"]), int(training["patch_size"]), total_epochs, steps_per_epoch, total_iters,
    )

    started_at = time()
    print_freq, val_freq, save_freq = int(training.get("print_freq", 20)), int(training.get("val_freq", 1000)), int(training.get("save_freq", 1000))
    try:
        for epoch in range(start_epoch, total_epochs + 1):
            model.train()
            epoch_start_step = resume_step if epoch == start_epoch else 0
            epoch_loader = create_train_loader(train_dataset, batch_size, workers, seed, epoch, epoch_start_step)
            for resumed_step, batch in enumerate(epoch_loader, start=1):
                step = epoch_start_step + resumed_step
                lq, gt = batch["lq"].to(device, non_blocking=True), batch["gt"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    restored_y, restored_uv, restored = model(lq)
                    loss_y = char_criterion(restored_y, gt[:, :1]) + ssim_criterion(restored_y, gt[:, :1])
                    loss_uv = char_criterion(restored_uv, gt[:, 1:]) + ssim_criterion(restored_uv, gt[:, 1:])
                    loss_yuv = char_criterion(restored, gt) + ssim_criterion(restored, gt)
                    weighted_y, weighted_uv = 0.01 * loss_y, 0.1 * loss_uv
                    total_loss = weighted_y + weighted_uv + loss_yuv
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                global_iter += 1
                elapsed = elapsed_since(started_at, previous_elapsed)
                eta = elapsed / global_iter * (total_iters - global_iter)
                writer.add_scalar("train/total_loss", total_loss.item(), global_iter)
                writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_iter)

                if global_iter % print_freq == 0:
                    train_logger.info(
                        "[%s][TRAIN] [progress: epoch=%d/%d, iter=%s/%s, step=%d/%d] [time: elapsed=%s, eta=%s] [optim: lr=%.3e] [total_loss: %.4f] [loss: y_weighted=%.4f, uv_weighted=%.4f, yuv=%.4f]",
                        experiment_name, epoch, total_epochs, f"{global_iter:,}", f"{total_iters:,}", step, steps_per_epoch,
                        format_duration(elapsed), format_duration(eta), optimizer.param_groups[0]["lr"], total_loss.item(),
                        weighted_y.item(), weighted_uv.item(), loss_yuv.item(),
                    )

                if global_iter % val_freq == 0 or global_iter == total_iters:
                    val_psnr, val_ssim = run_validation(model, val_loader, device)
                    updated = val_psnr > best_psnr
                    if updated:
                        best_psnr, best_rgb_ssim = val_psnr, val_ssim
                        save_generator_weights(model, models_dir / "best_G.pth")
                    writer.add_scalar("val/psnr", val_psnr, global_iter)
                    writer.add_scalar("val/rgb_ssim", val_ssim, global_iter)
                    val_logger.info(
                        "[%s][VAL] [progress: epoch=%d/%d, iter=%s/%s] [data: name=%s/%s] [metric: psnr=%.4f, rgb_ssim=%.4f] [best: key=psnr, value=%.4f, rgb_ssim=%s, updated=%s]",
                        experiment_name, epoch, total_epochs, f"{global_iter:,}", f"{total_iters:,}", dataset_config["name"],
                        dataset_config["val_split"], val_psnr, val_ssim, best_psnr,
                        f"{best_rgb_ssim:.4f}" if best_rgb_ssim is not None else "unknown", "yes" if updated else "no",
                    )

                if global_iter % save_freq == 0 or global_iter == total_iters:
                    elapsed = elapsed_since(started_at, previous_elapsed)
                    save_generator_weights(model, models_dir / "latest_G.pth")
                    save_generator_weights(model, models_dir / f"{global_iter}_G.pth")
                    save_training_state(
                        state_dir / f"{global_iter}.state",
                        make_state(model, optimizer, scheduler, scaler, epoch, step, global_iter, elapsed, best_psnr, best_rgb_ssim),
                    )
                    train_logger.info("[%s][CHECKPOINT] saved iter=%d", experiment_name, global_iter)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
