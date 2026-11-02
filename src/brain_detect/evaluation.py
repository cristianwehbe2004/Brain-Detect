"""Held-out segmentation evaluation and report metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from torch.utils.data import DataLoader


def segmentation_metrics(prediction: np.ndarray, target: np.ndarray, num_classes: int) -> dict:
    """Calculate per-class Dice, sensitivity, and specificity."""
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    metrics = {}
    for class_index in range(1, num_classes):
        predicted = prediction == class_index
        actual = target == class_index
        true_positive = np.logical_and(predicted, actual).sum()
        false_positive = np.logical_and(predicted, ~actual).sum()
        false_negative = np.logical_and(~predicted, actual).sum()
        true_negative = np.logical_and(~predicted, ~actual).sum()
        denominator = 2 * true_positive + false_positive + false_negative
        metrics[str(class_index)] = {
            "dice": float(2 * true_positive / denominator) if denominator else 1.0,
            "sensitivity": float(true_positive / (true_positive + false_negative))
            if true_positive + false_negative
            else 1.0,
            "specificity": float(true_negative / (true_negative + false_positive))
            if true_negative + false_positive
            else 1.0,
        }
    foreground_dice = [value["dice"] for value in metrics.values()]
    metrics["mean_foreground"] = {
        "dice": float(np.mean(foreground_dice)),
        "sensitivity": float(np.mean([value["sensitivity"] for value in metrics.values()])),
        "specificity": float(np.mean([value["specificity"] for value in metrics.values()])),
    }
    return metrics


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    roi_size: tuple[int, int, int] = (96, 96, 96),
    overlap: float = 0.25,
) -> dict:
    """Evaluate a model on a deterministic loader without modifying its state."""
    model.eval()
    patient_metrics = []
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"][:, 0].cpu().numpy()
        logits = sliding_window_inference(images, roi_size, 1, model, overlap=overlap)
        predictions = logits.argmax(dim=1).cpu().numpy()
        for prediction, target in zip(predictions, labels):
            patient_metrics.append(segmentation_metrics(prediction, target, logits.shape[1]))
    if not patient_metrics:
        raise ValueError("Test loader produced no batches")
    class_keys = [key for key in patient_metrics[0] if key != "mean_foreground"]
    result = {
        "patients_evaluated": len(patient_metrics),
        "per_class": {
            key: {
                metric: float(np.mean([patient[key][metric] for patient in patient_metrics]))
                for metric in ("dice", "sensitivity", "specificity")
            }
            for key in class_keys
        },
        "mean_foreground": {
            metric: float(np.mean([patient["mean_foreground"][metric] for patient in patient_metrics]))
            for metric in ("dice", "sensitivity", "specificity")
        },
    }
    return result


def load_model_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> dict:
    """Load model weights and return checkpoint metadata."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint
