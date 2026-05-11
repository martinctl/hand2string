"""CLI entry point for the improved retrieval training pipeline.

Supports single-GPU, multi-GPU (torchrun), and SLURM (sbatch) runs.

Single GPU:
    python scripts/train_improved.py --config configs/transformer.yaml

Multi-GPU (torchrun):
    torchrun --nproc_per_node=2 scripts/train_improved.py --config configs/transformer.yaml

Via SLURM:
    sbatch submit_job.sh configs/transformer.yaml <WANDB_KEY> 2
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.training.train import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/transformer.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    train(str(config_path))


if __name__ == "__main__":
    main()
