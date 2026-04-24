"""CLI: run MediaPipe over a dataset and cache landmark sequences."""
import argparse

from src.preprocessing.mediapipe_extractor import extract_from_video


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="dataset root or video path")
    parser.add_argument("--output", required=True, help="where to write .npz files")
    args = parser.parse_args()
    extract_from_video(args.input)  # TODO: iterate, write to args.output


if __name__ == "__main__":
    main()
