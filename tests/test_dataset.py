from pathlib import Path

import nibabel as nib
import numpy as np

from brain_detect.data.dataset import create_loader
from brain_detect.data.decathlon import DecathlonRecord, validate_labels


def make_record(tmp_path: Path) -> DecathlonRecord:
    image = np.ones((8, 8, 8, 4), dtype=np.float32)
    label = np.zeros((8, 8, 8), dtype=np.uint8)
    label[2:6, 2:6, 2:6] = 1
    image[2:6, 2:6, 2:6, 0] = 10
    affine = np.eye(4)
    image_path = tmp_path / "image.nii.gz"
    label_path = tmp_path / "label.nii.gz"
    nib.save(nib.Nifti1Image(image, affine), image_path)
    nib.save(nib.Nifti1Image(label, affine), label_path)
    return DecathlonRecord("case", image_path, label_path)


def test_validate_labels_and_validation_loader(tmp_path: Path) -> None:
    record = make_record(tmp_path)

    assert validate_labels(record, {0, 1}) == {0, 1}
    batch = next(iter(create_loader([record], training=False, num_workers=0)))

    assert tuple(batch["image"].shape) == (1, 4, 96, 96, 96)
    assert tuple(batch["label"].shape) == (1, 1, 96, 96, 96)
    assert set(np.unique(batch["label"].numpy())) == {0, 1}


def test_training_loader_collates_random_crops(tmp_path: Path) -> None:
    record = make_record(tmp_path)

    batch = next(iter(create_loader([record], training=True, num_workers=0)))

    assert tuple(batch["image"].shape) == (1, 4, 96, 96, 96)
    assert tuple(batch["label"].shape) == (1, 1, 96, 96, 96)
