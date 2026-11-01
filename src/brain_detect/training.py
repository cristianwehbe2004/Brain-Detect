"""Training and evaluation utilities for the segmentation baseline."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from torch.optim import AdamW
from torch.utils.data import DataLoader

from brain_detect.models.unet import create_unet


@dataclass(frozen=True)
class TrainingConfig:
    """Parameters recorded with every baseline experiment."""

    seed: int = 42
    epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 1
    patch_size: tuple[int, int, int] = (96, 96, 96)
    num_classes: int = 4
    validation_overlap: float = 0.25


def set_seed(seed: int) -> None:
    """Set reproducible random seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_labels(labels: torch.Tensor) -> torch.Tensor:
    """Convert MONAI label batches to integer class indices."""
    return labels.long()


def _one_hot_labels(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert class-index labels to channel-first one-hot tensors."""
    return torch.nn.functional.one_hot(labels[:, 0], num_classes=num_classes).permute(0, 4, 1, 2, 3).float()


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: torch.nn.Module,
    device: torch.device,
) -> float:
    """Train one epoch and return mean batch loss."""
    model.train()
    losses = []
    for batch in loader:
        images = batch["image"].to(device)
        labels = _prepare_labels(batch["label"].to(device))
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(images), labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    if not losses:
        raise ValueError("Training loader produced no batches")
    return float(np.mean(losses))


@torch.no_grad()
def validate_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_function: torch.nn.Module,
    device: torch.device,
    roi_size: tuple[int, int, int],
    overlap: float,
) -> tuple[float, float]:
    """Evaluate loss and mean foreground Dice using sliding-window inference."""
    model.eval()
    loss_values = []
    metric = DiceMetric(include_background=False, reduction="mean")
    for batch in loader:
        images = batch["image"].to(device)
        labels = _prepare_labels(batch["label"].to(device))
        logits = sliding_window_inference(images, roi_size, 1, model, overlap=overlap)
        loss_values.append(float(loss_function(logits, labels).cpu()))
        metric(
            y_pred=_one_hot_labels(logits.argmax(dim=1, keepdim=True), logits.shape[1]),
            y=_one_hot_labels(labels, logits.shape[1]),
        )
    if not loss_values:
        raise ValueError("Validation loader produced no batches")
    dice = float(metric.aggregate().cpu())
    metric.reset()
    return float(np.mean(loss_values)), dice


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: TrainingConfig,
    history: list[dict[str, float]],
) -> None:
    """Save model state and report metadata in one portable checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "history": history,
        },
        path,
    )


def train_baseline(
    train_loader: DataLoader,
    validation_loader: DataLoader,
    config: TrainingConfig,
    output_dir: Path,
    device: torch.device | None = None,
) -> list[dict[str, float]]:
    """Train the U-Net baseline and save best/latest checkpoints and history."""
    set_seed(config.seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_unet(out_channels=config.num_classes).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    history: list[dict[str, float]] = []
    best_dice = float("-inf")

    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_function, device)
        validation_loss, validation_dice = validate_epoch(
            model,
            validation_loader,
            loss_function,
            device,
            config.patch_size,
            config.validation_overlap,
        )
        result = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_dice": validation_dice,
        }
        history.append(result)
        save_checkpoint(output_dir / "latest.pt", model, optimizer, epoch, config, history)
        if validation_dice > best_dice:
            best_dice = validation_dice
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, config, history)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return history
