import numpy as np

from brain_detect.evaluation import segmentation_metrics


def test_segmentation_metrics_perfect_prediction() -> None:
    target = np.zeros((4, 4, 4), dtype=np.int64)
    target[1:3, 1:3, 1:3] = 1

    metrics = segmentation_metrics(target, target, num_classes=2)

    assert metrics["1"]["dice"] == 1.0
    assert metrics["mean_foreground"]["sensitivity"] == 1.0
    assert metrics["mean_foreground"]["specificity"] == 1.0


def test_segmentation_metrics_empty_class_is_well_defined() -> None:
    empty = np.zeros((4, 4, 4), dtype=np.int64)

    metrics = segmentation_metrics(empty, empty, num_classes=2)

    assert metrics["1"]["dice"] == 1.0
