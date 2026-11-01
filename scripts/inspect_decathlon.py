"""Inspect Medical Segmentation Decathlon Task01_BrainTumour."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brain_detect.data.decathlon import (
    check_alignment,
    discover_records,
    intensity_statistics,
    patient_splits,
    validate_metadata,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/decathlon/Task01_BrainTumour"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/decathlon_inspection.json"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metadata = validate_metadata(args.data_root)
    records = discover_records(args.data_root)
    if not records:
        raise SystemExit(f"No matching training records found under {args.data_root}")
    alignment = [check_alignment(record) for record in records]
    misaligned = [report for report in alignment if not report["aligned"]]
    if misaligned:
        raise SystemExit(f"Found {len(misaligned)} misaligned image/label pairs")
    report = {
        "name": metadata.get("name"),
        "modalities": metadata["modality"],
        "labels": metadata["labels"],
        "training_cases": len(records),
        "splits": patient_splits(records, seed=args.seed),
        "alignment": alignment,
        "statistics": {record.patient_id: intensity_statistics(record) for record in records},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Validated {len(records)} training case(s). Wrote {args.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
