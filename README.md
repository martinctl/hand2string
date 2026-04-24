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
