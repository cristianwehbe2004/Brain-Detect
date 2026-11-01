"""Inspect and validate a local BraTS 2021 dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brain_detect.data.brats import (
    check_alignment,
    discover_patients,
    intensity_statistics,
    patient_splits,
    validate_completeness,
    write_json,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = discover_patients(args.data_root)
    if not records:
        raise SystemExit(f"No patient directories with NIfTI files found under {args.data_root}")
    if not args.allow_incomplete:
        validate_completeness(records)

    complete_records = [record for record in records if record.complete]
    alignment = [check_alignment(record).__dict__ for record in complete_records]
    misaligned = [report for report in alignment if not report["aligned"]]
    if misaligned and not args.allow_incomplete:
        raise SystemExit(f"Found {len(misaligned)} patients with misaligned images or masks")

    write_manifest(records, args.output_dir / "manifest.csv")
    write_json(patient_splits(complete_records, seed=args.seed), args.output_dir / "splits.json")
    write_json(
        {
            record.patient_id: intensity_statistics(record)
            for record in complete_records
            if not any(report["patient_id"] == record.patient_id for report in misaligned)
        },
        args.output_dir / "dataset_statistics.json",
    )
    print(f"Discovered {len(records)} patient(s); validated {len(complete_records)} complete patient(s).")
    print(f"Wrote outputs to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
