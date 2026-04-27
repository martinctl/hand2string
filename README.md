# hand2string

Real-Time Sign Language Translation: A Landmark-Based Approach for ASL Recognition.

CS-503 Visual Intelligence project (EPFL, MA4).

## Pipeline

1. **MediaPipe landmark extraction** — 3D hand/body coordinates per frame.
2. **Lightweight classifier** — MLP (static signs) or LSTM/1D-CNN (dynamic gestures) over landmark sequences.
3. **LLM post-processing** — maps raw ASL word sequences to fluent English.

## Repository layout

```
hand2string/
├── data/                   # datasets (gitignored)
├── configs/                # training / experiment configs
├── notebooks/              # exploration
├── scripts/                # CLI entry points (download, preprocess, train, run)
└── src/
    ├── dataset/            # dataset download & loaders (ASL Alphabet, WLASL, How2Sign)
    ├── preprocessing/      # MediaPipe landmark extraction
    ├── models/             # MLP, LSTM, 1D-CNN
    ├── training/           # training loop, window-size sweep
    ├── evaluation/         # Precision/Recall/F1, latency (FPS), BLEU
    ├── llm/                # ASL → English translator (local Llama / API)
    └── inference/          # live webcam + foveated cropping
```

## Datasets

- [ASL Alphabet](https://www.kaggle.com/datasets/dorukdemirci/asl-alphabet-dataset) — PoC (static).
- [WLASL](https://github.com/dxli94/WLASL) — 2,000 signs, 100+ signers.
- [How2Sign](https://how2sign.github.io/) — 80h continuous ASL with English transcripts.

## Get the dataset

Sentence-level How2Sign clips are mirrored on the Hub at
[`martinctl/how2sign-asl-clips`](https://huggingface.co/datasets/martinctl/how2sign-asl-clips):

```python
from huggingface_hub import snapshot_download
import pandas as pd
from pathlib import Path

local = Path(snapshot_download("martinctl/how2sign-asl-clips", repo_type="dataset"))
df    = pd.read_parquet(local / "metadata.parquet")
clip  = local / df.iloc[0].file_name      # playable mp4
print(df.iloc[0].sentence)
```

On slurm, set `HF_HOME=/scratch/$USER/hf_cache` once and the call hits a
shared cache.

For an end-to-end demo (download → MediaPipe Holistic → skeleton overlay +
subtitle), see `examples/visualize_one_sentence.py`.

### Rebuilding / extending the dataset

```bash
# 1. cut clips locally from a downloaded shard
python scripts/build_hf_dataset.py \
    --csv ../how2sign_realigned_train.csv \
    --videos-dir ../shard_001_083 \
    --out data/how2sign_hf

# 2. push to the Hub (HF_TOKEN must be in hand2string/.env)
python scripts/upload_hf_dataset.py \
    --local data/how2sign_hf \
    --repo-id martinctl/how2sign-asl-clips
```

## Setup

```bash
conda env create -f environment.yml
conda activate hand2string
```

Or, if updating an existing env:

```bash
conda env update -f environment.yml --prune
```

## Team

Antoine Gautier, Coralie Banuls, Martin Catheland, Rached Toukko, Louis Larher.
