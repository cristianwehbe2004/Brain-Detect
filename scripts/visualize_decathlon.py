"""Save a central-slice multi-modal image and label visualization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from brain_detect.data.decathlon import discover_records, load_record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/decathlon/Task01_BrainTumour"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/decathlon_example.png"))
    parser.add_argument("--case", type=int, default=0)
    args = parser.parse_args()

    records = discover_records(args.data_root)
    if not records:
        raise SystemExit("No training cases found")
    if not 0 <= args.case < len(records):
        raise SystemExit(f"--case must be between 0 and {len(records) - 1}")
    image, label = load_record(records[args.case])
    slice_index = image.shape[2] // 2
    fig, axes = plt.subplots(1, image.shape[3], figsize=(16, 4), constrained_layout=True)
    for channel, axis in enumerate(np.atleast_1d(axes)):
        axis.imshow(np.rot90(image[:, :, slice_index, channel]), cmap="gray")
        axis.imshow(np.rot90(label[:, :, slice_index]), cmap="nipy_spectral", alpha=0.35, vmin=0)
        axis.set_title(f"Channel {channel}")
        axis.axis("off")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"Saved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
