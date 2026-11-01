from __future__ import annotations

from itertools import chain
from pathlib import Path

import pytest

from brain_detect.data.brats import (
    MODALITIES,
    PatientRecord,
    discover_patients,
    patient_splits,
    validate_completeness,
)


def make_patient(root: Path, patient_id: str, missing: str | None = None) -> None:
    patient_dir = root / patient_id
    patient_dir.mkdir()
    for kind in (*MODALITIES, "seg"):
        if kind != missing:
            (patient_dir / f"{patient_id}_{kind}.nii.gz").touch()


def test_discovers_required_brats_files(tmp_path: Path) -> None:
    make_patient(tmp_path, "BraTS2021_00000")

    records = discover_patients(tmp_path)

    assert len(records) == 1
    assert records[0].complete
    assert set(records[0].files) == {*MODALITIES, "seg"}


def test_completeness_reports_missing_files(tmp_path: Path) -> None:
    make_patient(tmp_path, "BraTS2021_00000", missing="t2")

    with pytest.raises(ValueError, match="missing t2"):
        validate_completeness(discover_patients(tmp_path))


def test_patient_splits_are_deterministic_and_disjoint() -> None:
    records = [PatientRecord(f"patient-{index}", {}) for index in range(10)]

    first = patient_splits(records, seed=7)
    second = patient_splits(records, seed=7)

    assert first == second
    assert set(first["train"]).isdisjoint(first["validation"])
    assert set(first["train"]).isdisjoint(first["test"])
    assert set(first["validation"]).isdisjoint(first["test"])
    assert sorted(chain.from_iterable(first.values())) == [f"patient-{index}" for index in range(10)]
