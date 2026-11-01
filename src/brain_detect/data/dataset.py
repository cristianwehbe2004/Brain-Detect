"""Model-ready MONAI datasets and loaders for Decathlon."""

from __future__ import annotations

from collections.abc import Sequence

from monai.data import Dataset
from torch.utils.data import DataLoader

from brain_detect.data.decathlon import DecathlonRecord
from brain_detect.data.transforms import training_transforms, validation_transforms


def records_to_items(records: Sequence[DecathlonRecord]) -> list[dict[str, str]]:
    """Convert records to the dictionary format expected by MONAI."""
    return [{"image": str(record.image), "label": str(record.label), "patient_id": record.patient_id} for record in records]


def create_dataset(
    records: Sequence[DecathlonRecord],
    training: bool = True,
    patch_size: tuple[int, int, int] = (96, 96, 96),
) -> Dataset:
    """Create a MONAI dataset with paired image and label transforms."""
    transforms = (
        training_transforms(patch_size)
        if training
        else validation_transforms()
    )
    return Dataset(data=records_to_items(records), transform=transforms)


def create_loader(
    records: Sequence[DecathlonRecord],
    training: bool = True,
    patch_size: tuple[int, int, int] = (96, 96, 96),
    batch_size: int = 1,
    num_workers: int = 0,
) -> DataLoader:
    """Create a PyTorch loader for model-ready samples."""
    dataset = create_dataset(records, training=training, patch_size=patch_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=training, num_workers=num_workers)
