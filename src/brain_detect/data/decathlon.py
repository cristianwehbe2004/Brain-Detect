"""Medical Segmentation Decathlon brain-tumour dataset utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np


@dataclass(frozen=True)
class DecathlonRecord:
    """One image volume and its corresponding segmentation label."""

    patient_id: str
    image: Path
    label: Path


def discover_records(dataset_root: Path) -> list[DecathlonRecord]:
    """Discover matching imagesTr and labelsTr files."""
    images_dir = dataset_root / "imagesTr"
    labels_dir = dataset_root / "labelsTr"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"Expected imagesTr and labelsTr under {dataset_root}")

    records = []
    for image in sorted(path for path in images_dir.glob("*.nii.gz") if not path.name.startswith(".")):
        patient_id = image.name.removesuffix(".nii.gz")
        label = labels_dir / f"{patient_id}.nii.gz"
        if label.is_file():
            records.append(DecathlonRecord(patient_id, image, label))
    return records


def validate_metadata(dataset_root: Path) -> dict:
    """Validate the required dataset metadata and return it."""
    metadata_path = dataset_root / "dataset.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing dataset metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required = {"modality", "labels", "training"}
    missing = required - set(metadata)
    if missing:
        raise ValueError(f"Dataset metadata is missing: {', '.join(sorted(missing))}")
    return metadata


def check_alignment(record: DecathlonRecord, atol: float = 1e-5) -> dict:
    """Check that image and label shapes and affines match."""
    image = nib.load(record.image)
    label = nib.load(record.label)
    mismatches = []
    if image.shape[:3] != label.shape:
        mismatches.append(f"shape {image.shape[:3]} != {label.shape}")
    if not np.allclose(image.affine, label.affine, atol=atol):
        mismatches.append("affine differs")
    return {
        "patient_id": record.patient_id,
        "image_shape": image.shape,
        "label_shape": label.shape,
        "aligned": not mismatches,
        "mismatches": mismatches,
    }


def load_record(record: DecathlonRecord) -> tuple[np.ndarray, np.ndarray]:
    """Load image channels as (C, H, W, D) and label as (H, W, D)."""
    image = np.asarray(nib.load(record.image).dataobj)
    label = np.asarray(nib.load(record.label).dataobj)
    if image.ndim != 4 or label.ndim != 3:
        raise ValueError(f"Unexpected dimensions for {record.patient_id}: {image.shape}, {label.shape}")
    return image, label


def validate_labels(record: DecathlonRecord, allowed_labels: set[int] | None = None) -> set[int]:
    """Return labels present in a mask and reject non-integer or unknown values."""
    label = np.asarray(nib.load(record.label).dataobj)
    if not np.all(np.isfinite(label)) or not np.all(label == np.rint(label)):
        raise ValueError(f"{record.patient_id} contains non-integer or non-finite labels")
    present = {int(value) for value in np.unique(label)}
    if allowed_labels is not None:
        unexpected = present - allowed_labels
        if unexpected:
            raise ValueError(f"{record.patient_id} contains unexpected labels: {sorted(unexpected)}")
    return present


def intensity_statistics(record: DecathlonRecord) -> dict[str, dict[str, float | int]]:
    """Calculate non-zero voxel statistics for every image channel."""
    image, _ = load_record(record)
    statistics = {}
    for channel in range(image.shape[3]):
        values = image[..., channel].astype(np.float64)
        values = values[np.isfinite(values) & (values != 0)]
        if values.size == 0:
            raise ValueError(f"{record.patient_id}/channel-{channel} has no valid voxels")
        statistics[str(channel)] = {
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
    records: list[DecathlonRecord],
    seed: int = 42,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> dict[str, list[str]]:
    """Create deterministic patient-level train/validation/test splits."""
    if validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("Validation and test fractions must be non-negative and sum to less than 1")
    patient_ids = np.array(sorted(record.patient_id for record in records))
    if len(patient_ids) < 2:
        raise ValueError("At least two patients are required for train/validation splits")
    np.random.default_rng(seed).shuffle(patient_ids)
    test_count = round(len(patient_ids) * test_fraction)
    validation_count = round(len(patient_ids) * validation_fraction)
    if validation_fraction > 0 and validation_count == 0:
        validation_count = 1
    if test_fraction > 0 and test_count == 0 and len(patient_ids) - validation_count >= 3:
        test_count = 1
    while test_count + validation_count >= len(patient_ids):
        if test_count >= validation_count and test_count > 0:
            test_count -= 1
        else:
            validation_count -= 1
    return {
        "train": patient_ids[test_count + validation_count :].tolist(),
        "validation": patient_ids[test_count : test_count + validation_count].tolist(),
        "test": patient_ids[:test_count].tolist(),
    }
