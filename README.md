# Strawberry Plant Health Classification

Code for training image classifiers that predict strawberry plant health
classes from image tiles. The repository intentionally excludes private
images, annotations, GPS/ground-truth tables, model checkpoints, videos, and
training outputs.

## What is included

- `3.1_train_strawberry_mc.py`: multiclass image-classification training
- `21_semisupervised.py` and `22_semisupervised_learning.py`: semisupervised
  classification experiments
- `5.strawberry_contrastive.py`: contrastive classification experiment
- `2.plant_health.py`: segmentation/pretraining experiment
- `4.build_image_level_csvs.py`: dataset-preparation utility
- `plot_loss.py`: loss-plotting utility
- Jupyter notebooks containing earlier experiments

The object-detection and segmentation-only scripts, datasets, checkpoints, and
generated media are excluded by `.gitignore`.

## Requirements

Use Python 3.10 or newer. A virtual environment is recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The checked-in requirements file was exported from a development environment
and includes PyTorch CPU packages. For GPU training, install the PyTorch build
appropriate for your CUDA version before installing the remaining packages.

## Dataset layout

The classification trainer expects a private `class_tiles` directory with
either of these layouts:

```text
class_tiles/
  train/
    healthy/
      images/*.jpg
    unhealthy/
      images/*.jpg
  val/
    healthy/
      images/*.jpg
    unhealthy/
      images/*.jpg
```

or:

```text
class_tiles/
  train/<class_name>/*.jpg
  val/<class_name>/*.jpg
```

Do not commit the dataset. Place it outside the repository or keep it ignored
by `.gitignore`.

## Train a classifier

After preparing `class_tiles`, provide its path explicitly:

```bash
python 3.1_train_strawberry_mc.py \
  --tiles_root /path/to/class_tiles \
  --out_dir /path/to/training_runs \
  --model resnet18 \
  --epochs 20 \
  --batch_size 32 \
  --pretrained
```

Supported models are `resnet18`, `resnet34`, `resnet50`, and `efficientnet_b0`.
Run the help command to see all options:

```bash
python 3.1_train_strawberry_mc.py --help
```

Training checkpoints and logs should be written outside the repository or
under ignored output directories.

## Important limitations

This repository is a code snapshot, not a fully self-contained reproducible
training package. The private dataset and trained checkpoints are not
included, so a complete training or inference run cannot be verified from a
fresh clone alone.

Some legacy scripts contain machine-specific default paths. Always pass their
input and output paths explicitly, and review a script before running it on a
different machine. The scripts were syntax-checked, but the complete data
pipeline and model training were not run here because the required private
data is excluded.

## Validation

Basic checks can be run with:

```bash
python -m compileall \
  2.plant_health.py \
  3.1_train_strawberry_mc.py \
  4.build_image_level_csvs.py \
  5.strawberry_contrastive.py \
  21_semisupervised.py \
  22_semisupervised_learning.py \
  plot_loss.py
```

## Data and model ownership

Confirm that you have permission to publish any source code, images, labels,
pretrained weights, and derived results before adding them to a public
repository.
