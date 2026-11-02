"""Evaluate a saved U-Net checkpoint on the held-out Decathlon test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brain_detect.data.dataset import create_loader
from brain_detect.data.decathlon import discover_records, patient_splits
from brain_detect.evaluation import evaluate_model, load_model_checkpoint
from brain_detect.models.unet import create_unet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    records = discover_records(args.data_root)
    if args.max_cases is not None:
        records = records[: args.max_cases]
    split_ids = patient_splits(records)
    by_id = {record.patient_id: record for record in records}
    test_records = [by_id[patient_id] for patient_id in split_ids["test"]]
    if not test_records:
        raise SystemExit("The selected dataset has no test patients; use more than two cases.")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("config", {})
    patch_size = tuple(config.get("patch_size", (96, 96, 96)))
    loader = create_loader(test_records, training=False, num_workers=args.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_unet(out_channels=int(config.get("num_classes", 4))).to(device)
    load_model_checkpoint(model, args.checkpoint, device)
    result = {
        "checkpoint": str(args.checkpoint),
        "split": "test",
        "split_counts": {key: len(value) for key, value in patient_splits(records).items()},
        "metrics": evaluate_model(model, loader, device, roi_size=patch_size),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Evaluated {result['metrics']['patients_evaluated']} test patient(s). Wrote {args.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
