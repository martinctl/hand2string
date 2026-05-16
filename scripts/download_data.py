"""CLI: download and prepare a dataset into data/."""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.dataset.download import download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", choices=["asl_alphabet", "wlasl", "how2sign", "how2sign_landmarks"], required=True)
    parser.add_argument("--root", default="data")
    args = parser.parse_args()
    download(args.name, args.root)


if __name__ == "__main__":
    main()
