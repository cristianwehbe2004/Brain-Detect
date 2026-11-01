"""MONAI transforms for model-ready brain-tumour volumes."""

from __future__ import annotations

from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    ScaleIntensityRangePercentilesd,
    SpatialPadd,
)


def training_transforms(patch_size: tuple[int, int, int] = (96, 96, 96)) -> Compose:
    """Build paired image/label transforms for training."""
    return Compose(
        [
            LoadImaged(keys=("image", "label")),
            EnsureChannelFirstd(keys=("image", "label")),
            Orientationd(keys=("image", "label"), axcodes="RAS", labels=None),
            CropForegroundd(keys=("image", "label"), source_key="image"),
            ScaleIntensityRangePercentilesd(
                keys="image",
                lower=1,
                upper=99,
                b_min=0,
                b_max=1,
                clip=True,
                channel_wise=True,
            ),
            SpatialPadd(keys=("image", "label"), spatial_size=patch_size),
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


def validation_transforms() -> Compose:
    """Build deterministic transforms for full-volume validation samples."""
    return Compose(
        [
            LoadImaged(keys=("image", "label")),
            EnsureChannelFirstd(keys=("image", "label")),
            Orientationd(keys=("image", "label"), axcodes="RAS", labels=None),
            CropForegroundd(keys=("image", "label"), source_key="image"),
            ScaleIntensityRangePercentilesd(
                keys="image",
                lower=1,
                upper=99,
                b_min=0,
                b_max=1,
                clip=True,
                channel_wise=True,
            ),
            SpatialPadd(keys=("image", "label"), spatial_size=(96, 96, 96)),
            EnsureTyped(keys="image", dtype=None),
            EnsureTyped(keys="label", dtype=None),
        ]
    )
