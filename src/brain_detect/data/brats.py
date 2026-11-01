"""BraTS 2021 dataset discovery and validation."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import nibabel as nib
import numpy as np

MODALITIES = ("flair", "t1", "t1ce", "t2")
SEGMENTATION = "seg"


@dataclass(frozen=True)
class PatientRecord:
    """Files belonging to one BraTS patient."""

    patient_id: str
    files: dict[str, Path]

    @property
    def complete(self) -> bool:
        return all(name in self.files for name in (*MODALITIES, SEGMENTATION))


@dataclass(frozen=True)
class AlignmentReport:
    """Spatial metadata comparison for one patient."""

    patient_id: str
    shape: tuple[int, ...]
    aligned: bool
    mismatches: tuple[str, ...]


def discover_patients(data_root: Path) -> list[PatientRecord]:
    """Find patient directories containing BraTS-style NIfTI files."""
    data_root = data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {data_root}")

    records: list[PatientRecord] = []
    for patient_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        files: dict[str, Path] = {}
        for path in sorted(patient_dir.glob("*.nii*")):
            parts = path.name.split("_")
            if len(parts) < 2:
                continue
            kind = parts[-1].removesuffix(".nii.gz").removesuffix(".nii")
            if kind in (*MODALITIES, SEGMENTATION):
                files[kind] = path
        if files:
            records.append(PatientRecord(patient_dir.name, files))
    return records


def validate_completeness(records: list[PatientRecord]) -> None:
    """Raise a useful error when any patient lacks a required file."""
    incomplete = [
        f"{record.patient_id}: missing {', '.join(name for name in (*MODALITIES, SEGMENTATION) if name not in record.files)}"
        for record in records
        if not record.complete
    ]
    if incomplete:
        raise ValueError("Incomplete BraTS patients:\n" + "\n".join(incomplete))


def load_patient(record: PatientRecord) -> tuple[np.ndarray, np.ndarray]:
    """Load four modalities as (C, H, W, D) and the mask as (H, W, D)."""
    if not record.complete:
        raise ValueError(f"Cannot load incomplete patient: {record.patient_id}")
    modalities = [np.asarray(nib.load(record.files[name]).dataobj) for name in MODALITIES]
    mask = np.asarray(nib.load(record.files[SEGMENTATION]).dataobj)
    return np.stack(modalities, axis=0), mask


def check_alignment(record: PatientRecord, atol: float = 1e-5) -> AlignmentReport:
    """Check that all modalities and the mask share shape and affine."""
    if not record.complete:
        raise ValueError(f"Cannot check incomplete patient: {record.patient_id}")
    images = {name: nib.load(path) for name, path in record.files.items()}
    reference = images[MODALITIES[0]]
    mismatches: list[str] = []
    for name, image in images.items():
        if image.shape != reference.shape:
            mismatches.append(f"{name}: shape {image.shape} != {reference.shape}")
        if not np.allclose(image.affine, reference.affine, atol=atol):
            mismatches.append(f"{name}: affine differs from {MODALITIES[0]}")
    return AlignmentReport(record.patient_id, tuple(reference.shape), not mismatches, tuple(mismatches))


def intensity_statistics(record: PatientRecord) -> dict[str, dict[str, Union[float, int]]]:  # noqa: UP007
    """Calculate non-zero voxel statistics for each MRI modality."""
    images, _ = load_patient(record)
    statistics: dict[str, dict[str, Union[float, int]]] = {}  # noqa: UP007
    for index, name in enumerate(MODALITIES):
        values = images[index].astype(np.float64)
        values = values[np.isfinite(values) & (values != 0)]
        if values.size == 0:
            raise ValueError(f"{record.patient_id}/{name} contains no finite non-zero voxels")
        statistics[name] = {
            "count": int(values.size),
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "p01": float(np.percentile(values, 1)),
            "p99": float(np.percentile(values, 99)),
        }
    return statistics


def patient_splits(
    records: list[PatientRecord],
    seed: int = 42,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> dict[str, list[str]]:
    """Create deterministic patient-level train/validation/test splits."""
    if validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must be non-negative and sum to less than 1")
    patient_ids = np.array(sorted(record.patient_id for record in records))
    rng = np.random.default_rng(seed)
    rng.shuffle(patient_ids)
    test_count = round(len(patient_ids) * test_fraction)
    validation_count = round(len(patient_ids) * validation_fraction)
    return {
        "train": patient_ids[test_count + validation_count :].tolist(),
        "validation": patient_ids[test_count : test_count + validation_count].tolist(),
        "test": patient_ids[:test_count].tolist(),
    }


def write_manifest(records: list[PatientRecord], output_path: Path) -> None:
    """Write a portable CSV manifest with relative file paths."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["patient_id", *MODALITIES, SEGMENTATION])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {"patient_id": record.patient_id, **{name: str(path) for name, path in record.files.items()}}
            )


def write_json(value: object, output_path: Path) -> None:
    """Write JSON with stable formatting."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def iter_alignment_reports(records: list[PatientRecord]) -> Iterator[AlignmentReport]:
    """Yield alignment reports for complete patients."""
    for record in records:
        yield check_alignment(record)
