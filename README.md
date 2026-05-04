# hand2string

Real-Time Sign Language Translation: a landmark-based approach for ASL video
understanding.

CS-503 Visual Intelligence project (EPFL, MA4).

## Current state

The repository now has two working baselines:

1. **ASL Alphabet PoC**: static hand landmarks -> MLP classifier.
2. **How2Sign sentence retrieval**: dynamic landmark time series -> matching
   English subtitle retrieval.

The second baseline is the main path for continuous signing. It does not try to
generate free-form English yet. Instead, it learns whether a video clip and a
subtitle sentence belong together. This gives us a trainable video-to-language
objective with the current How2Sign sentence-level annotations.

## Dynamic retrieval pipeline

1. **Clip dataset**: How2Sign RGB frontal clips with English subtitles.
2. **MediaPipe extraction**: one `.npz` per clip with:
   - `landmarks`: `(T, 75, 3)` for `33 pose + 21 left hand + 21 right hand`
   - `mask`: `(T, 75)` detection mask for missing landmarks
   - subtitle/id/split metadata
3. **Landmark transform**:
   - torso-center and scale each frame
   - zero-fill missing landmarks after normalization
   - concatenate xyz + mask
   - resample each clip to `128` frames
   - default model input: `(128, 75 * 4) = (128, 300)`
4. **Two-tower model**:
   - video tower: BiGRU over landmark frames
   - text tower: frozen `sentence-transformers/all-MiniLM-L6-v2` embeddings,
     projected into the shared retrieval space
   - loss: symmetric contrastive loss over in-batch video/text pairs
5. **Evaluation**:
   - validation top-1/top-k retrieval
   - qualitative top-k subtitle inspection with `scripts/evaluate_retrieval.py`

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

## Setup

```bash
conda env create -f environment.yml
conda activate hand2string
```

If the environment already exists:

```bash
conda env update -f environment.yml --prune
```

`sentence-transformers<5` is pinned because v5 currently pulls a TorchCodec
import path that can break in this environment. The default MiniLM model is
downloaded from Hugging Face on first use and then loaded from the local cache.

## Datasets

- [ASL Alphabet](https://www.kaggle.com/datasets/dorukdemirci/asl-alphabet-dataset) — PoC (static).
- [WLASL](https://github.com/dxli94/WLASL) — 2,000 signs, 100+ signers.
- [How2Sign](https://how2sign.github.io/) — 80h continuous ASL with English transcripts.

## Download How2Sign clips

Sentence-level How2Sign clips are mirrored on the Hub at
[`martinctl/how2sign-asl-clips`](https://huggingface.co/datasets/martinctl/how2sign-asl-clips):

```bash
python scripts/download_data.py --name how2sign --root data
```

Hugging Face may store the dataset under
`data/datasets--martinctl--how2sign-asl-clips/snapshots/<hash>/`. The extractor
resolves this automatically, so `--input data`, `--input data/how2sign_hf`, and
the real snapshot path all work.

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

## ASL Alphabet PoC (Phase 1)

Static-sign baseline: MediaPipe HandLandmarker -> MLP classifier on
[ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) (29 classes).

```bash
# 1. download (requires Kaggle API token in ~/.kaggle/kaggle.json)
kaggle datasets download -d grassknoted/asl-alphabet -p data/asl_alphabet --unzip

# 2. extract hand landmarks (-> data/asl_alphabet_landmarks.npz)
python scripts/extract_alphabet_landmarks.py \
    --root data/asl_alphabet/asl_alphabet_train/asl_alphabet_train \
    --out data/asl_alphabet_landmarks.npz

# 3. train MLP baseline (~2 min on CPU, ~98% val acc)
python scripts/train_mlp.py --data data/asl_alphabet_landmarks.npz --epochs 30

# 4. live webcam demo
python scripts/live_alphabet.py --ckpt runs/mlp_alphabet/best.pt
```

## How2Sign sentence-retrieval baseline

The default config in `configs/default.yaml` trains the MiniLM retrieval
baseline.

```bash
# 1. Extract landmarks. Add --limit N for quick experiments.
python scripts/extract_landmarks.py \
    --input data/how2sign_hf \
    --output data/how2sign_landmarks \
    --split train \
    --target-fps 25

# 2. Train the retrieval model.
python scripts/train.py --config configs/default.yaml

# 3. Inspect validation retrievals.
python scripts/evaluate_retrieval.py \
    --ckpt runs/how2sign_retrieval/best.pt \
    --top-k 5 \
    --max-queries 10
# Optional: save all top-k rows for later error analysis.
python scripts/evaluate_retrieval.py \
    --ckpt runs/how2sign_retrieval/best.pt \
    --out-csv runs/how2sign_retrieval/retrieval_val.csv
```

Useful config switches:

- `training.device: auto`: CUDA -> Apple Silicon MPS -> CPU fallback.
- `training.device: cuda | mps | cpu`: force a specific backend.
- `preprocessing.landmark_layout: full`: all 75 landmarks.
- `preprocessing.landmark_layout: upper`: pose `0-16` plus both hands.
- `text.encoder: sentence_transformer`: default MiniLM text tower.
- `text.encoder: tfidf`: dependency-light control baseline.

The first small run on 347 cached clips produced a non-random retrieval signal:
random top-1 over 69 validation clips is about `0.0145`, while the TF-IDF
baseline reached about `0.1449` top-1. MiniLM is now the default next baseline
to compare on the same split and on larger caches.

## Evaluation output

`scripts/evaluate_retrieval.py` reports aggregate retrieval metrics and prints
qualitative examples:

```text
Retrieval eval: 69 queries | device=cpu | top1=... top5=... median_rank=...

query 1/69 | id=... | true_rank=...
GT: ...
* 01. score=... id=... :: retrieved subtitle
  02. score=... id=... :: retrieved subtitle
```

Use this before over-tuning. The examples reveal whether the model learns
motion/sign content or shortcuts from neighboring clips and repeated topics.

## Next steps

1. **Qualitative retrieval review**: run `scripts/evaluate_retrieval.py` and
   inspect the top-5 subtitles for good and bad examples. Add `--out-csv
   runs/how2sign_retrieval/retrieval_val.csv` to save all ranks.
2. **Scale the landmark cache**: extract more clips before heavy tuning. Use
   `--limit 1000` for a medium smoke run, then remove `--limit` for the full
   dataset.
3. **Ablate landmark layouts**: compare `preprocessing.landmark_layout: full`
   against `upper` to test whether lower-body pose points add useful signal.
4. **Use stronger validation splits**: split by `video_id` once more shards are
   available, so neighboring clips from the same source video do not leak into
   both train and validation.
5. **Compare text encoders**: keep MiniLM as the default semantic text tower,
   but run `text.encoder: tfidf` as a lightweight control experiment.
6. **Move to GPU for full runs**: keep local CPU/MPS for smoke tests, then run
   `training.device: cuda` on SLURM or another CUDA machine for the full cache.
7. **Later translation model**: once retrieval is stable, use the video encoder
   as the visual front-end for sequence generation or for gloss/word-level
   decoding if better annotations become available.

## Team

Antoine Gautier, Coralie Banuls, Martin Catheland, Rached Toukko, Louis Larher.
