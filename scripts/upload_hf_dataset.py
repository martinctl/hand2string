"""Upload a built dataset folder to the Hugging Face Hub.

Usage:
    python scripts/upload_hf_dataset.py \
        --local data/how2sign_hf \
        --repo-id martinctl/how2sign-asl-clips [--private]

    python scripts/upload_hf_dataset.py \
        --local data/how2sign_landmarks_hf \
        --repo-id martinctl/how2sign-asl-landmarks \
        --commit-message "Upload How2Sign landmark shards"
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=Path, required=True, help="dataset folder to upload")
    parser.add_argument("--repo-id", required=True, help="e.g. martinctl/how2sign-asl-clips")
    parser.add_argument("--private", action="store_true", help="create private (default: public)")
    parser.add_argument("--commit-message", default="Update How2Sign dataset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(repo_root / ".env")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not found in environment or hand2string/.env")

    if not (args.local / "metadata.parquet").exists():
        raise SystemExit(f"{args.local} has no metadata.parquet - run build_hf_dataset.py first")

    api = HfApi(token=token)

    me = api.whoami()["name"]
    print(f"Authenticated as: {me}")

    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    print(f"Repo ready: https://huggingface.co/datasets/{args.repo_id}")

    api.upload_folder(
        folder_path=str(args.local),
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=args.commit_message,
        ignore_patterns=[".*", "**/.*", "*.pyc", "__pycache__/**"],
    )
    print("Upload complete.")


if __name__ == "__main__":
    main()
