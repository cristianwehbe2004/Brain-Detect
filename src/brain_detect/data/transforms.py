"""MONAI transforms for model-ready brain-tumour volumes."""

from __future__ import annotations

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    ScaleIntensityRanged,
)


def training_transforms(patch_size: tuple[int, int, int] = (96, 96, 96)) -> Compose:
    """Build paired image/label transforms for training."""
    return Compose(
        [
            LoadImaged(keys=("image", "label")),
            EnsureChannelFirstd(keys=("image", "label")),
            Orientationd(keys=("image", "label"), axcodes="RAS"),
            ScaleIntensityRanged(
                keys="image",
                a_min=0,
                a_max=500,
                b_min=0,
                b_max=1,
                clip=True,
            ),
            RandCropByPosNegLabeld(
                keys=("image", "label"),
                label_key="label",
                spatial_size=patch_size,
                pos=1,
                neg=1,
                num_samples=1,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=("image", "label"), spatial_axis=0, prob=0.5),
            RandRotate90d(keys=("image", "label"), prob=0.5, max_k=3),
            EnsureTyped(keys=("image", "label")),
        ]
    )
