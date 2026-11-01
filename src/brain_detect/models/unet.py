"""Compact 3D U-Net baseline for four-channel Decathlon MRI."""

from __future__ import annotations

from monai.networks.nets import UNet


def create_unet(
    in_channels: int = 4,
    out_channels: int = 4,
    channels: tuple[int, ...] = (16, 32, 64, 128),
    strides: tuple[int, ...] = (2, 2, 2),
) -> UNet:
    """Create the reproducible baseline architecture."""
    if len(channels) != len(strides) + 1:
        raise ValueError("channels must contain exactly one more value than strides")
    return UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=2,
    )
