"""Audit Decathlon images, labels, alignment, and split integrity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brain_detect.data.decathlon import (
    check_alignment,
    discover_records,
    load_record,
    patient_splits,
    validate_labels,
    validate_metadata,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()

    metadata = validate_metadata(args.data_root)
    records = discover_records(args.data_root)
    if args.max_cases is not None:
        records = records[: args.max_cases]
    allowed_labels = {int(key) for key in metadata["labels"]}
    cases = []
    for record in records:
        image, _ = load_record(record)
        alignment = check_alignment(record)
        present_labels = validate_labels(record, allowed_labels)
        cases.append(
            {
                "patient_id": record.patient_id,
                "channels": image.shape[3],
                "finite_image": bool(np.isfinite(image).all()),
                "alignment": alignment,
                "labels": sorted(present_labels),
            }
        )
    splits = patient_splits(records)
    all_ids = [patient_id for values in splits.values() for patient_id in values]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Patient split overlap detected")
    report = {
        "case_count": len(records),
        "metadata_labels": metadata["labels"],
        "splits": {key: len(value) for key, value in splits.items()},
        "all_aligned": all(case["alignment"]["aligned"] for case in cases),
        "all_finite": all(case["finite_image"] for case in cases),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Audited {len(records)} case(s). Wrote {args.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
