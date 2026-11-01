from pathlib import Path

import torch

from brain_detect.models.unet import create_unet
from brain_detect.training import TrainingConfig, save_checkpoint


def test_unet_produces_four_class_logits() -> None:
    model = create_unet(channels=(2, 4, 8, 16))

    output = model(torch.rand(1, 4, 32, 32, 32))

    assert tuple(output.shape) == (1, 4, 32, 32, 32)


def test_checkpoint_contains_report_metadata(tmp_path: Path) -> None:
    model = create_unet(channels=(2, 4, 8, 16))
    optimizer = torch.optim.AdamW(model.parameters())
    config = TrainingConfig(epochs=2)
    path = tmp_path / "checkpoint.pt"

    save_checkpoint(path, model, optimizer, 1, config, [{"epoch": 1.0, "validation_dice": 0.2}])
    checkpoint = torch.load(path, map_location="cpu")

    assert checkpoint["epoch"] == 1
    assert checkpoint["config"]["epochs"] == 2
    assert checkpoint["history"][0]["validation_dice"] == 0.2
