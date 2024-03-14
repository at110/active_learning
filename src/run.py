import argparse
from src import train, predict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spleen Segmentation Task")
    parser.add_argument("--mode", type=str, choices=["train", "predict"], help="Run mode: train or predict.")
    args = parser.parse_args()

    if args.mode == "train":
        train.main()
    elif args.mode == "predict":
        predict.main()