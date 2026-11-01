"""Visualize a saved U-Net prediction against a Decathlon label."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from monai.inferers import sliding_window_inference

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brain_detect.data.dataset import create_loader
from brain_detect.data.decathlon import discover_records
from brain_detect.models.unet import create_unet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/decathlon/Task01_BrainTumour"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed/unet_prediction.png"))
    parser.add_argument("--case", type=int, default=0)
    args = parser.parse_args()

    records = discover_records(args.data_root)
    if not 0 <= args.case < len(records):
        raise SystemExit(f"--case must be between 0 and {len(records) - 1}")
    loader = create_loader([records[args.case]], training=False, batch_size=1)
    batch = next(iter(loader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_unet(out_channels=4).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        prediction = sliding_window_inference(
            batch["image"].to(device), (96, 96, 96), 1, model, overlap=0.25
        ).argmax(dim=1)[0].cpu()
    image = batch["image"][0, 0].cpu()
    label = batch["label"][0, 0].cpu()
    slice_index = image.shape[2] // 2
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, data, title in zip(
        axes,
        (image, label, prediction),
        ("MRI channel 0", "Ground truth", "Prediction"),
    ):
        axis.imshow(data[:, :, slice_index].T, cmap="gray" if "MRI" in title else "nipy_spectral")
        axis.set_title(title)
        axis.axis("off")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
