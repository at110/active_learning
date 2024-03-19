import argparse
from src import train, predict
import os
import torch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.transforms import Compose, AsDiscrete
from monai.data import decollate_batch
from typing import Tuple, List, Dict
from model import build_model  # Adjust the import path as necessary
from data_loader import create_data_loaders, get_post_transforms_unlabelled ,create_data_loaders_predictions # Adjust the import path as necessary
import mlflow
import json
import numpy as np
from scipy.stats import entropy
from typing import List, Tuple
import nibabel as nib
from monai.handlers.utils import from_engine
import shutil
from torch.utils.data import DataLoader
from torch.optim import Optimizer
import subprocess
from torch.nn import Module
#from torch.device import Device

def load_config(config_path: str = 'config.json') -> Dict:
    """
    Load configuration from a JSON file.

    Parameters:
        config_path: The path to the configuration file.

    Returns:
        A dictionary containing the configuration.
    """
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    return config


def setup_mlflow(config: Dict) -> None:
    """
    Set up MLflow tracking and experiments based on the configuration.

    Parameters:
        config: A dictionary containing the MLflow configuration.
    """
    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    mlflow.set_experiment(config["experiment_name"])
    mlflow.start_run()
    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
    mlflow.log_params({
        "learning_rate": config["model_params"]["learning_rate"],
        "batch_size": config["model_params"]["batch_size"]
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spleen Segmentation Task")
    parser.add_argument("--mode", type=str, choices=["train", "predict"], help="Run mode: train or predict.")
    args = parser.parse_args()
    config = load_config()
    setup_mlflow(config)

    if args.mode == "train":
        train.main()
    elif args.mode == "predict":
        predict.main()
    
    mlflow.end_run()