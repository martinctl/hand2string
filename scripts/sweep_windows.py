"""CLI: run the window-size Pareto sweep."""
import argparse

from src.training.sweep import run_sweep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    run_sweep(args.config)


if __name__ == "__main__":
    main()
