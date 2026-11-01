from pathlib import Path

import nibabel as nib
import numpy as np

from brain_detect.data.decathlon import (
    DecathlonRecord,
    check_alignment,
    discover_records,
    load_record,
    patient_splits,
)


def make_dataset(root: Path) -> None:
    (root / "imagesTr").mkdir()
    (root / "labelsTr").mkdir()
    image = np.zeros((4, 5, 6, 4), dtype=np.float32)
    label = np.zeros((4, 5, 6), dtype=np.uint8)
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(image, affine), root / "imagesTr" / "brain_000.nii.gz")
    nib.save(nib.Nifti1Image(label, affine), root / "labelsTr" / "brain_000.nii.gz")
    (root / "dataset.json").write_text(
        '{"modality": {"0": "MRI"}, "labels": {"0": "background"}, "training": []}',
        encoding="utf-8",
    )


def test_discovers_and_loads_decathlon_record(tmp_path: Path) -> None:
    make_dataset(tmp_path)

    records = discover_records(tmp_path)
    image, label = load_record(records[0])

    assert records[0].patient_id == "brain_000"
    assert image.shape == (4, 5, 6, 4)
    assert label.shape == (4, 5, 6)
    assert check_alignment(records[0])["aligned"]


def test_patient_splits_are_deterministic() -> None:
    records = [DecathlonRecord(f"brain_{index:03d}", Path("image"), Path("label")) for index in range(10)]

    assert patient_splits(records, seed=3) == patient_splits(records, seed=3)
