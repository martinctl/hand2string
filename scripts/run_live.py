"""CLI: live webcam inference."""
import argparse

from src.inference.live import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
