import json
from pathlib import Path

import nibabel as nib
import numpy as np

from brain_detect.data.decathlon import discover_records


def test_decathlon_fixture_has_four_channels(tmp_path: Path) -> None:
    (tmp_path / "imagesTr").mkdir()
    (tmp_path / "labelsTr").mkdir()
    image = np.zeros((4, 4, 4, 4), dtype=np.float32)
    label = np.zeros((4, 4, 4), dtype=np.uint8)
    nib.save(nib.Nifti1Image(image, np.eye(4)), tmp_path / "imagesTr" / "case.nii.gz")
    nib.save(nib.Nifti1Image(label, np.eye(4)), tmp_path / "labelsTr" / "case.nii.gz")
    (tmp_path / "dataset.json").write_text(
        json.dumps(
            {
                "modality": {"0": "MRI-0", "1": "MRI-1", "2": "MRI-2", "3": "MRI-3"},
                "labels": {"0": "background", "1": "tumor"},
                "training": [],
            }
        ),
        encoding="utf-8",
    )

    records = discover_records(tmp_path)

    assert len(records) == 1
