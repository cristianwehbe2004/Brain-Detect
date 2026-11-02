# Brain-Detect Project Guide

This document explains the project for new contributors.

## 1. Project goal

Brain-Detect is a research project for multi-modal brain MRI tumor
segmentation. The model receives four MRI channels and predicts a
voxel-level tumor segmentation mask.

The long-term research direction is an attention-enhanced hybrid architecture
combining:

1. CNN layers for local spatial patterns.
2. Transformer blocks for global context.
3. Attention-based feature fusion.

We are not starting with the hybrid model. A conventional 3D U-Net is the
baseline. The hybrid model must use the same data splits, preprocessing, and
metrics so that comparisons are scientifically meaningful.

## 2. Dataset

The current dataset is the Medical Segmentation Decathlon
`Task01_BrainTumour`, not BraTS 2021. It contains four MRI channels for each
case and a corresponding segmentation mask:

```text
Task01_BrainTumour/
├── dataset.json
├── imagesTr/
│   ├── BRATS_001.nii.gz
│   └── ...
├── labelsTr/
│   ├── BRATS_001.nii.gz
│   └── ...
└── imagesTs/
```

The dataset is large and must not be committed to GitHub. It is also medical
research data, so it should only be shared through an approved, private
storage location.

The official archive source is:

```text
https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar
```

## 3. Repository structure

```text
src/brain_detect/data/decathlon.py
    Dataset discovery, loading, alignment, label validation, and splits.

src/brain_detect/data/transforms.py
    MONAI training and validation preprocessing transforms.

src/brain_detect/data/dataset.py
    MONAI Dataset and PyTorch DataLoader factories.

src/brain_detect/models/unet.py
    Compact 3D U-Net baseline.

src/brain_detect/training.py
    Training loop, validation loop, Dice loss, checkpoints, and history.

src/brain_detect/evaluation.py
    Held-out test metrics: Dice, sensitivity, and specificity.

scripts/audit_decathlon.py
    Dataset integrity and split audit.

scripts/train_unet.py
    U-Net training entry point.

scripts/evaluate_test.py
    Held-out test evaluation entry point.

scripts/visualize_prediction.py
    Ground-truth and prediction visualization.

scripts/plot_history.py
    Training and validation curve plot.
```

## 4. Data preparation and preprocessing

Raw NIfTI files remain unchanged. Preprocessing is applied dynamically when a
sample is loaded:

1. Load the image and label NIfTI files.
2. Ensure channel-first tensors.
3. Reorient paired images and labels to RAS.
4. Crop irrelevant foreground background.
5. Normalize each MRI channel using percentile intensity scaling.
6. Pad volumes to at least `96 x 96 x 96`.
7. During training, sample positive/negative 3D patches of
   `96 x 96 x 96`.
8. Apply training-only random flips and rotations.
9. Keep image and label spatial transformations paired.
10. Use integer labels and nearest-neighbor-compatible label handling.

Training returns tensors shaped approximately as:

```text
image: (batch, 4, 96, 96, 96)
label: (batch, 1, 96, 96, 96)
```

Validation and test transforms are deterministic and do not use random
augmentation.

## 5. Patient-level splitting

Cases are split by patient identifier, never by individual slices or patches.
The deterministic default is:

```text
60% train
20% validation
20% test
seed = 42
```

The test set is held out. It must not be used to select epochs, tune
hyperparameters, or compare architecture variants.

## 6. Baseline training

The baseline is a four-channel 3D U-Net with four output classes:

```text
4 MRI channels
        ↓
3D U-Net encoder/decoder
        ↓
background + tumor label logits
```

The training loop uses:

- AdamW optimizer
- Dice plus Cross Entropy loss
- Sliding-window validation inference
- Foreground Dice for model selection
- `best.pt` and `latest.pt` checkpoints
- `history.json` for reportable epoch metrics
- `config.json` for reproducibility

Run a short smoke test:

```bash
python scripts/train_unet.py \
  --data-root /path/to/Task01_BrainTumour \
  --output-dir experiments/unet_smoke_test \
  --max-cases 10 \
  --epochs 1 \
  --num-workers 0
```

Run a real baseline experiment:

```bash
python scripts/train_unet.py \
  --data-root /path/to/Task01_BrainTumour \
  --output-dir experiments/unet_baseline \
  --epochs 5 \
  --num-workers 0
```

The current 20-case, 2-epoch experiment is only a development smoke test:
its best validation Dice was approximately `0.0264`. It should not be
presented as the final model result.

## 7. Evaluation and report artifacts

Audit the dataset before a reportable run:

```bash
python scripts/audit_decathlon.py \
  --data-root /path/to/Task01_BrainTumour \
  --output experiments/dataset_audit.json
```

Evaluate the selected checkpoint on the held-out test patients:

```bash
python scripts/evaluate_test.py \
  --data-root /path/to/Task01_BrainTumour \
  --checkpoint experiments/unet_baseline/best.pt \
  --output experiments/unet_baseline/test_metrics.json \
  --num-workers 0
```

Generate report figures:

```bash
python scripts/plot_history.py \
  experiments/unet_baseline/history.json \
  --output experiments/unet_baseline/training_curves.png

python scripts/visualize_prediction.py \
  --data-root /path/to/Task01_BrainTumour \
  --checkpoint experiments/unet_baseline/best.pt \
  --output experiments/unet_baseline/prediction.png
```

Important report artifacts are:

```text
config.json
history.json
test_metrics.json
training_curves.png
prediction.png
best.pt
```

## 8. Local development setup

Clone the repository and install the project:

```bash
git clone https://github.com/cristianwehbe2004/Brain-Detect.git
cd Brain-Detect
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
```

Contributors should create a branch:

```bash
git checkout -b feature/short-description
```

Do not commit datasets, checkpoints, or generated experiment directories.
Open a pull request after running tests and lint checks.

## 9. Recommended Colab workflow

Colab is used for GPU computation; GitHub is used for source code; Google
Drive is used for persistent datasets and experiment outputs.

### Start a Colab session

```python
from google.colab import drive
drive.mount("/content/drive")
```

Clone or update the code:

```python
!rm -rf /content/Brain-Detect
!git clone https://github.com/cristianwehbe2004/Brain-Detect.git /content/Brain-Detect
%cd /content/Brain-Detect
!pip install -e ".[dev]" --quiet
```

Copy the dataset from Drive to Colab local storage before training:

```python
!rm -rf /content/Task01_BrainTumour
!cp -r \
  "/content/drive/MyDrive/Brain-Detect/data/decathlon/Task01_BrainTumour" \
  "/content/Task01_BrainTumour"
```

Use `/content/Task01_BrainTumour` for `--data-root`. Reading thousands of
compressed NIfTI files directly from mounted Drive can cause slowdowns or
transport disconnection errors.

Save checkpoints and reports back to Drive:

```python
!python scripts/train_unet.py \
  --data-root "/content/Task01_BrainTumour" \
  --output-dir "/content/drive/MyDrive/Brain-Detect/experiments/unet_baseline" \
  --epochs 5 \
  --num-workers 0
```

## 10. How to share the dataset with a collaborator

Do not upload the dataset to GitHub. Choose one of these approaches:

### Option A: Shared Google Drive folder

1. Create a private Google Drive folder for the project.
2. Share it with the collaborator's Google account as **Viewer** or
   **Editor**, according to your needs.
3. Put the extracted `Task01_BrainTumour` directory in that folder.
4. The collaborator mounts their Drive in Colab.
5. They copy the dataset to `/content` before training.

This is the simplest option when both contributors can use Google Drive.

### Option B: Each collaborator downloads the official archive

Give the collaborator the official archive URL and have them download it in
their own Colab session or approved storage. This avoids copying a large
dataset between accounts.

### Option C: Shared cloud storage

For a larger team, use an institution-approved shared bucket or storage
workspace. Give collaborators only the access they need and follow the
dataset's terms of use.

The collaborator does not need the dataset in the GitHub repository. They
only need the repository code plus a local/Colab `--data-root` path.

## 11. Collaboration responsibilities

Suggested division of work:

- Contributor A: data audit, preprocessing, and loader changes.
- Contributor B: baseline training, evaluation, and experiment tracking.
- Contributor C: hybrid CNN-Transformer architecture and ablations.
- Everyone: tests, documentation, and review.

Each experiment should have a unique directory. Do not overwrite another
person's checkpoint or metrics. Record the Git commit, dataset split seed,
configuration, hardware, runtime, and results with every experiment.

