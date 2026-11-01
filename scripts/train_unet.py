"""Train the reproducible 3D U-Net baseline on Decathlon."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brain_detect.data.dataset import create_loader
from brain_detect.data.decathlon import discover_records, patient_splits
from brain_detect.training import TrainingConfig, train_baseline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/decathlon/Task01_BrainTumour"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/experiments/unet_baseline"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    records = discover_records(args.data_root)
    if args.max_cases is not None:
        records = records[: args.max_cases]
    split_ids = patient_splits(records)
    by_id = {record.patient_id: record for record in records}
    train_records = [by_id[patient_id] for patient_id in split_ids["train"]]
    validation_records = [by_id[patient_id] for patient_id in split_ids["validation"]]
    config = TrainingConfig(epochs=args.epochs)
    train_loader = create_loader(
        train_records,
        training=True,
        patch_size=config.patch_size,
        batch_size=config.batch_size,
        num_workers=args.num_workers,
    )
    validation_loader = create_loader(
        validation_records,
        training=False,
        patch_size=config.patch_size,
        batch_size=1,
        num_workers=args.num_workers,
    )
    history = train_baseline(
        train_loader,
        validation_loader,
        config,
        args.output_dir,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )
    print(f"Completed {len(history)} epoch(s). Best checkpoint: {args.output_dir / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
