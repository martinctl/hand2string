"""Cut every CSV row whose source video is present locally into its own mp4
and assemble an HF-friendly dataset folder ready for upload.

Usage:
    python scripts/build_hf_dataset.py \
        --csv ../how2sign_realigned_train.csv \
        --videos-dir ../shard_001_083 \
        --out data/how2sign_hf \
        --split train [--limit N]
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.dataset.clipping import cut_clip


DATASET_README = """\
---
license: cc-by-nc-4.0
task_categories:
  - video-classification
  - translation
language:
  - en
tags:
  - sign-language
  - asl
  - how2sign
size_categories:
  - n<10K
---

# how2sign-asl-clips

Sentence-level clips from the [How2Sign](https://how2sign.github.io/) ASL
dataset, cut to the realigned timestamps in `how2sign_realigned_train.csv`.
Built for the EPFL CS-503 *hand2string* project.

This is a **work-in-progress** mirror of a single shard ({n_videos} source
videos, {n_clips} clips). More splits and shards will be added as we
download them.

## Schema (`metadata.parquet`)

| column         | type   | notes                                       |
|----------------|--------|---------------------------------------------|
| sentence_id    | string | primary key, e.g. `--7E2sU6zP4_10`          |
| sentence_name  | string | full How2Sign clip name with camera tag     |
| video_id       | string | parent YouTube id                           |
| video_name     | string | source mp4 basename (no extension)          |
| start          | float  | timestamp in source video (s)               |
| end            | float  | timestamp in source video (s)               |
| duration       | float  | `end - start` (s)                           |
| sentence       | string | English transcript                          |
| split          | string | dataset split                               |
| file_name      | string | clip path relative to repo root             |

## Quick start

```python
from huggingface_hub import snapshot_download
import pandas as pd
from pathlib import Path

local = Path(snapshot_download("martinctl/how2sign-asl-clips", repo_type="dataset"))
df    = pd.read_parquet(local / "metadata.parquet")

row = df.iloc[0]
print(row.sentence)
clip = local / row.file_name   # playable mp4
```

## License

How2Sign is released under **CC BY-NC 4.0**. Cite the original authors:

> Duarte, A., Palaskar, S., Ventura, L., Ghadiyaram, D., DeHaan, K., Metze, F.,
> Torres, J., Giró-i-Nieto, X. *How2Sign: A Large-scale Multimodal Dataset for
> Continuous American Sign Language.* CVPR 2021.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="how2sign realigned CSV")
    parser.add_argument("--videos-dir", type=Path, required=True, help="folder of source mp4s")
    parser.add_argument("--out", type=Path, required=True, help="output dataset folder")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=None, help="only cut N rows (smoke test)")
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="parallel encode workers (default: min(8, cpu_count))",
    )
    return parser.parse_args()


def _cut_one(job: dict) -> dict:
    """Worker entry point. Returns a status dict so the parent can aggregate."""
    src = Path(job["src"])
    dst = Path(job["dst"])
    if dst.exists():
        return {"status": "skipped", "row": job["row"]}
    try:
        cut_clip(src=src, start=job["start"], end=job["end"], dst=dst, crf=job["crf"])
    except Exception as exc:
        return {"status": "failed", "name": job["row"]["sentence_name"], "error": repr(exc)}
    return {"status": "ok", "row": job["row"]}


def main() -> None:
    args = parse_args()
    out_root: Path = args.out
    clips_dir = out_root / "clips" / args.split
    clips_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv, sep="\t")
    needed = {"VIDEO_ID", "VIDEO_NAME", "SENTENCE_ID", "SENTENCE_NAME",
              "START_REALIGNED", "END_REALIGNED", "SENTENCE"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing required columns: {sorted(missing)}")

    available = {p.stem for p in args.videos_dir.glob("*.mp4")}
    df_present = df[df["VIDEO_NAME"].isin(available)].reset_index(drop=True)

    print(
        f"CSV rows: {len(df)} | source videos found: {len(available)} | "
        f"rows with a local source: {len(df_present)}"
    )

    if args.limit is not None:
        df_present = df_present.head(args.limit).copy()
        print(f"--limit {args.limit} -> processing {len(df_present)} rows")

    jobs: list[dict] = []
    for r in df_present.itertuples(index=False):
        sentence_name = r.SENTENCE_NAME
        clip_rel = Path("clips") / args.split / f"{sentence_name}.mp4"
        clip_abs = out_root / clip_rel
        row = {
            "sentence_id": r.SENTENCE_ID,
            "sentence_name": sentence_name,
            "video_id": r.VIDEO_ID,
            "video_name": r.VIDEO_NAME,
            "start": float(r.START_REALIGNED),
            "end": float(r.END_REALIGNED),
            "duration": float(r.END_REALIGNED) - float(r.START_REALIGNED),
            "sentence": r.SENTENCE,
            "split": args.split,
            "file_name": clip_rel.as_posix(),
        }
        jobs.append({
            "src": str(args.videos_dir / f"{r.VIDEO_NAME}.mp4"),
            "dst": str(clip_abs),
            "start": row["start"],
            "end": row["end"],
            "crf": args.crf,
            "row": row,
        })

    rows: list[dict] = []
    n_skipped_existing = 0
    n_failed = 0

    workers = max(1, args.workers)
    print(f"Encoding with {workers} worker process(es)")

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_cut_one, j) for j in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            res = fut.result()
            if res["status"] == "ok":
                rows.append(res["row"])
            elif res["status"] == "skipped":
                rows.append(res["row"])
                n_skipped_existing += 1
            else:
                n_failed += 1
                tqdm.write(f"FAIL {res['name']}: {res['error']}")

    if not rows:
        raise SystemExit("no clips produced; aborting")

    meta = pd.DataFrame(rows)
    meta_path = out_root / "metadata.parquet"
    if meta_path.exists():
        existing = pd.read_parquet(meta_path)
        meta = (
            pd.concat([existing, meta], ignore_index=True)
            .drop_duplicates(subset=["sentence_id", "split"], keep="last")
            .reset_index(drop=True)
        )
    meta.to_parquet(meta_path, index=False)

    readme_path = out_root / "README.md"
    readme_path.write_text(
        DATASET_README.format(
            n_videos=meta["video_name"].nunique(),
            n_clips=len(meta),
        ),
        encoding="utf-8",
    )

    total_bytes = sum(p.stat().st_size for p in clips_dir.glob("*.mp4"))
    print(
        f"\nDone.\n"
        f"  clips on disk: {len(list(clips_dir.glob('*.mp4')))}\n"
        f"  metadata rows: {len(meta)}\n"
        f"  skipped (already cut): {n_skipped_existing}\n"
        f"  failed: {n_failed}\n"
        f"  size: {total_bytes / 1e9:.2f} GB\n"
        f"  out: {out_root}"
    )


if __name__ == "__main__":
    main()
