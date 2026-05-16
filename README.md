# hand2string

Real-time sign language translation experiments for EPFL CS-503 Visual Intelligence.

The current shared training artifact is a preprocessed How2Sign landmark dataset on Hugging Face:

https://huggingface.co/datasets/martinctl/how2sign-asl-landmarks

It contains sentence-level MediaPipe landmarks, compact geometric features, validity masks, timestamps, source metadata, and documented failure accounting. It does not contain MP4 clips, so teammates do not need to download or preprocess the original How2Sign videos before training.

## Quick Start

Create the environment:

```bash
conda env create -f environment.yml
conda activate hand2string
```

Or update an existing environment:

```bash
conda env update -f environment.yml --prune
conda activate hand2string
```

Download the preprocessed landmark dataset:

```bash
python scripts/download_data.py --name how2sign_landmarks --root data
```

This creates:

```text
data/how2sign_landmarks/
  metadata.parquet
  failures.parquet
  feature_names.json
  preprocessing_config.json
  shards/*.npz
```

Run a loader smoke check:

```bash
python scripts/check_how2sign_landmark_dataset.py --root data/how2sign_landmarks --max-rows 1000
```

Run a tiny training smoke test:

```bash
python scripts/train_improved.py --config configs/how2sign_quickstart.yaml
```

For a real experiment, start from:

```bash
python scripts/train_improved.py --config configs/transformer.yaml
```

## Dataset Notes

The How2Sign source data has known text/video mismatch issues. We keep this explicit:

- `metadata.parquet` contains 35,176 successful landmark clips.
- `failures.parquet` contains 87 expected CSV rows that could not be generated from the local videos.
- `audit_report.json` summarizes the final dataset integrity check.

For training, use `metadata.parquet` only and keep the default `min_frames: 10` filter unless you specifically want to study very short segments.

The built-in loader supports three feature inputs:

- `landmarks`: flattened `(128 landmarks x 3)` image coordinates per frame
- `geometric`: 36 compact hand/body/face relation features per frame
- `landmarks+geometric`: concatenation of both

The current configs use `geometric` first because it is much lighter and easier to train/debug.

## Repository Layout

```text
configs/                    training configs
examples/                   visual inspection demos
scripts/                    download, dataset build, upload, train entry points
src/dataset/                dataset download and PyTorch loaders
src/models/                 retrieval/video/text model modules
src/preprocessing/          MediaPipe schema, feature recipes, dataset builder
src/training/               training loop, device selection, text encoders
data/                       local datasets, gitignored
runs/                       local training outputs, gitignored
```

## Useful Commands

Download the shared dataset:

```bash
python scripts/download_data.py --name how2sign_landmarks --root data
```

Inspect a feature dashboard from locally available raw videos:

```bash
python examples/how2sign_feature_dashboard.py \
  --dataset data/how2sign_landmarks \
  --videos-root .. \
  --split train \
  --out outputs/demo/how2sign_feature_dashboard.html
```

Rebuild the landmark dataset from local raw How2Sign videos:

```bash
python scripts/build_how2sign_landmark_dataset.py \
  --root .. \
  --out data/how2sign_landmarks_hf \
  --target-fps 25 \
  --pose-model lite \
  --workers 4 \
  --samples-per-shard 128
```

Push a rebuilt dataset to Hugging Face:

```bash
hf upload-large-folder martinctl/how2sign-asl-landmarks data/how2sign_landmarks_hf --repo-type dataset
```

## Team

Antoine Gautier, Coralie Banuls, Martin Catheland, Rached Toukko, Louis Larcher.
