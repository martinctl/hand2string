"""CLI: train a classifier from a config."""
import argparse

from src.training.train import train


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
