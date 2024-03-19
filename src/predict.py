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

import glob
from monai.data import CacheDataset, DataLoader
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandAffined,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    Invertd,
)


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

def save_prediction_as_nifti(
    model,
    loader, 
    post_transforms_unlabelled,
    device, 
    op_dir, 
    subdir, 
    root_dir,
    filenames
):
    """
    Saves model predictions as NIfTI files.

    Parameters:
        model: The trained model for generating predictions.
        loader: DataLoader for the dataset to predict.
        device: The device on which the model is loaded.
        output_directory: The directory for saving NIfTI files.
        filenames: Filenames for the output NIfTI files.
    """
    # Load the best saved model state
    post_pred = post_transforms_unlabelled
    os.makedirs(op_dir, exist_ok=True)
    os.makedirs(f'{op_dir}/{subdir}', exist_ok=True)
    print(device)
    model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth"), map_location=device))
    model.eval()
    model.to(device)
    subject_num = 0 
    with torch.no_grad():
        for data in loader:
           
            roi_size = (160, 160, 160)
            sw_batch_size = 4
            # Perform inference
            #val_outputs = sliding_window_inference(data["image"].to(device), roi_size, sw_batch_size, model)
            data["pred"] = sliding_window_inference(data["image"].to(device), roi_size, sw_batch_size, model)

            data = [post_pred(i) for i in decollate_batch(data)]
            predictions = from_engine(["pred"])(data)
            predictions = predictions[0].detach().cpu()[1, :, :, :]
            ## Convert the predictions tensor to int16 data type
            predictions = predictions.numpy().astype(np.int16)

            # Convert the predictions tensor to a NIfTI image
            pred_nifti = nib.Nifti1Image(predictions, affine=np.eye(4))

            # Save the NIfTI image to file
            filename = filenames[ subject_num]['image'].split('/')[-1].split('_')[1].split('.')[0]+'.nii.gz'
            nib.save(pred_nifti, os.path.join(f'{op_dir}/{subdir}',filename))
            print(f"Saved predictions as NIfTI file at: {os.path.join(f'{op_dir}/{subdir}', filename)}")
            subject_num+=1

def log_nifti_directory_as_artifacts(directory_path: str):
    """
    Logs all NIfTI files in a specified directory as MLflow artifacts.

    Parameters:
        directory_path: The directory containing NIfTI files to log.
    """
    mlflow.log_artifacts(directory_path, artifact_path=f'predictions')
    shutil.rmtree(directory_path)

def main():
    
    config = load_config()
    #setup_mlflow(config)

    #loaders_predictions = create_data_loaders_predictions(data_dir=config["data_loader_params"]["data_dir"], batch_size=1, num_workers=config["data_loader_params"]["num_workers"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    #save_prediction_as_nifti(model, loaders_predictions["val"],device, "./predictions","val", config["root_dir"], loaders_predictions["val_files"])
    #save_prediction_as_nifti(model, loaders_predictions["test"],device, "./predictions","test", config["root_dir"], loaders_predictions["test_files"])
    #save_prediction_as_nifti(model, loaders_predictions["train"],device, "./predictions","train", config["root_dir"], loaders_predictions["train_files"])
    #save_prediction_as_nifti(model, loaders_predictions["unlabelled"],device, "./predictions","unlabelled", config["root_dir"], loaders_predictions["unlabelled_files"])
    
    unlabelled_transforms = Compose(
    [
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-0,
            a_max=200,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image"),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode="bilinear"),
    ]
    )

    post_transforms_unlabelled = Compose(
    [
        Invertd(
            keys="pred",
            transform=unlabelled_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
            device="cpu",
            allow_missing_keys=True
        ),
        AsDiscreted(keys="pred", argmax=True, to_onehot=2),
        #AsDiscreted(keys="label", to_onehot=2),
    ]
    )

    unlabelled_images = sorted(glob.glob(os.path.join( "../Spleen-stratified/imagesUnlabelled", "*.nii.gz")))
    unlabelled_files = [{"image": img} for img in unlabelled_images]

    unlabelled_ds = CacheDataset(data=unlabelled_files, transform=unlabelled_transforms, cache_rate=1.0, num_workers=4)
    unlabelled_loader = DataLoader(unlabelled_ds, batch_size=1, num_workers=4)
    save_prediction_as_nifti(model, unlabelled_loader,post_transforms_unlabelled, device, "./predictions","val", config["root_dir"], unlabelled_files)

    #save_prediction_as_nifti(model, unlabelled_loader,post_transforms_unlabelled, device, "./predictions","unlabelled", config["root_dir"], unlabelled_files)
    # Log NIfTI directory as artifacts
    
    
    log_nifti_directory_as_artifacts("./predictions")

    #mlflow.end_run()

if __name__ == "__main__":
    main()
