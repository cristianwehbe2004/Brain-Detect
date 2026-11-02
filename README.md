# Brain Detect

Research workspace for multi-modal brain MRI tumor segmentation.

## Milestone 1: dataset preparation

The active development dataset is Medical Segmentation Decathlon
`Task01_BrainTumour`. It contains four MRI channels per case and a
segmentation label. The official archive is hosted by MONAI:

```text
https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar
```

The archive is stored under `data/decathlon/`, separate from the BraTS
directory, because its layout is different. Download and extract it with:

```bash
mkdir -p data/decathlon
curl -L --continue-at - \
  https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar \
  -o data/decathlon/Task01_BrainTumour.tar
tar -xf data/decathlon/Task01_BrainTumour.tar -C data/decathlon
```

Inspect, split, and visualize the cases:

```bash
python scripts/inspect_decathlon.py
python scripts/visualize_decathlon.py
```

The inspection report includes metadata, channel statistics, alignment
results, and deterministic patient-level train/validation/test splits.

## Colab workflow

To keep the 7.3 GB archive off your Mac, open a Google Colab notebook and run:

```python
!git clone <your-repository-url> /content/Brain-Detect
%cd /content/Brain-Detect
!pip install -e ".[dev]"
!mkdir -p data/decathlon
!curl -L --continue-at - \
  https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar \
  -o data/decathlon/Task01_BrainTumour.tar
!tar -xf data/decathlon/Task01_BrainTumour.tar -C data/decathlon
!python scripts/inspect_decathlon.py
!python scripts/visualize_decathlon.py
```

For persistence across Colab sessions, mount Google Drive and point
`--data-root` at a copy of the extracted `Task01_BrainTumour` directory.

The MONAI training transform is available from
`brain_detect.data.transforms.training_transforms`.
Model-ready datasets and PyTorch loaders are available from
`brain_detect.data.dataset.create_dataset` and `create_loader`. They crop
training samples to `96 x 96 x 96` patches, normalize each foreground
channel by percentiles, preserve paired image/label spatial transforms, and
pad validation volumes to the minimum patch size.

## 3D U-Net baseline

Train the baseline in Colab after the real-data loader check succeeds:

```bash
python scripts/train_unet.py \
  --data-root data/decathlon/Task01_BrainTumour \
  --output-dir data/experiments/unet_baseline \
  --epochs 5
```

For a smoke test, add `--max-cases 10 --epochs 1`. The experiment writes
`best.pt`, `latest.pt`, `history.json`, and `config.json`. Save that directory
to Google Drive for the project report. Visualize a prediction with:

```bash
python scripts/visualize_prediction.py \
  --data-root data/decathlon/Task01_BrainTumour \
  --checkpoint data/experiments/unet_baseline/best.pt
```

## Audit and held-out test evaluation

Run the preprocessing/data audit before a reportable experiment:

```bash
python scripts/audit_decathlon.py \
  --data-root data/decathlon/Task01_BrainTumour \
  --output data/experiments/dataset_audit.json
```

Evaluate the selected `best.pt` exactly once on the patient-level test split:

```bash
python scripts/evaluate_test.py \
  --data-root data/decathlon/Task01_BrainTumour \
  --checkpoint data/experiments/unet_baseline/best.pt \
  --output data/experiments/unet_baseline/test_metrics.json
```

The test report contains mean and per-class Dice, sensitivity, and specificity.
Generate report figures from the training history with:

```bash
python scripts/plot_history.py \
  data/experiments/unet_baseline/history.json \
  --output data/experiments/unet_baseline/training_curves.png
```

BraTS support remains available as a separate loader. BraTS data is
access-controlled and must be downloaded from an approved source after
accepting its terms. Place the extracted patient directories under
`data/raw/`, for example:

```text
data/raw/
└── BraTS2021_00000/
    ├── BraTS2021_00000_flair.nii.gz
    ├── BraTS2021_00000_seg.nii.gz
    ├── BraTS2021_00000_t1.nii.gz
    ├── BraTS2021_00000_t1ce.nii.gz
    └── BraTS2021_00000_t2.nii.gz
```

Do not commit the dataset. The included tooling validates that every patient
has all four modalities and a segmentation mask, checks shape/affine
alignment, reports intensity statistics, and creates deterministic
patient-level splits.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Inspect the dataset

```bash
python scripts/inspect_dataset.py --data-root data/raw --output-dir data/processed
```

The command writes `manifest.csv`, `dataset_statistics.json`, and
`splits.json` to the output directory. It exits with an error if any patient
is incomplete or has misaligned NIfTI files. Use `--allow-incomplete` only
when intentionally inventorying a partially extracted download.

## Current scope

This milestone does not train a model or download data automatically. The
next steps are visualization, preprocessing transforms, and model-ready
dataset objects after the local dataset has been verified.
