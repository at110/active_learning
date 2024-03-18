import os
import torch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.transforms import Compose, AsDiscrete
from monai.data import decollate_batch
from typing import Tuple, List, Dict
from model import build_model  # Adjust the import path as necessary
from data_loader import create_data_loaders, get_post_transforms_unlabelled, create_data_loaders_predictions # Adjust the import path as necessary
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





def train_epoch(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, 
                optimizer: torch.optim.Optimizer, loss_function: torch.nn.Module, 
                device: torch.device) -> float:
    """
    Train the model for one epoch.

    Parameters:
    - model: torch.nn.Module - The model being trained.
    - train_loader: DataLoader - DataLoader for training data.
    - optimizer: torch.optim.Optimizer - Optimizer for model parameters.
    - loss_function: torch.nn.Module - Loss function used for training.
    - device: torch.device - Device on which to train.

    Returns:
    - float: Average loss for this epoch.
    """
    model.train()
    epoch_loss = 0.0
    steps = 0
    for batch_data in train_loader:
        steps += 1
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / steps

def validate(model: torch.nn.Module, val_loader: torch.utils.data.DataLoader, 
             loss_function: torch.nn.Module, device: torch.device, 
             post_pred: Compose, post_label: Compose, dice_metric: DiceMetric) -> Tuple[float, float]:
    """
    Validate the model on the validation dataset.

    Parameters:
    - model: torch.nn.Module - The model being validated.
    - val_loader: DataLoader - DataLoader for validation data.
    - loss_function: torch.nn.Module - Loss function used for validation.
    - device: torch.device - Device on which to validate.
    - post_pred: Compose - Post-processing transformations for predictions.
    - post_label: Compose - Post-processing transformations for labels.
    - dice_metric: DiceMetric - Metric for evaluation.

    Returns:
    - Tuple[float, float]: Average validation loss and Dice metric for the validation dataset.
    """
    model.eval()
    val_loss = 0.0

    steps = 0
    with torch.no_grad():
        for batch_data in val_loader:
            steps += 1
            inputs, labels = (batch_data["image"].to(device), batch_data["label"].to(device))
            roi_size = (160, 160, 160)  # Adjust this to match your model's expected input dimensions
            sw_batch_size = 4  # Adjust based on your GPU memory
            # Assuming val_inputs is your input tensor shaped (batch_size, channels, D, H, W)
            # Ensure val_inputs has the correct shape, add batch dimension if necessary
            if len(inputs.shape) == 4:  # Missing batch dimension, assuming shape (channels, D, H, W)
                inputs = inputs.unsqueeze(0)  # Now shape (1, channels, D, H, W)
            outputs = sliding_window_inference(inputs, roi_size=(160, 160, 160), sw_batch_size=4, predictor=model)
            loss = loss_function(outputs, labels)
            val_loss += loss.item()
            val_outputs = [post_pred(i) for i in decollate_batch(outputs)]
            val_labels = [post_label(i) for i in decollate_batch(labels)]
            dice_metric(y_pred=val_outputs, y=val_labels)
    metric = dice_metric.aggregate().item()
    dice_metric.reset()
    return val_loss / steps, metric

def save_best_model(metric: float, best_metric: float, model: torch.nn.Module, epoch: int, 
                    root_dir: str) -> Tuple[float, int]:
    """
    Save the model if the current metric is better than the best metric.

    Parameters:
    - metric: float - Current metric value.
    - best_metric: float - Best metric value achieved so far.
    - model: torch.nn.Module - Model to save.
    - epoch: int - Current epoch number.
    - root_dir: str - Directory where the model will be saved.

    Returns:
    - Tuple[float, int]: Updated best metric and epoch.
    """
    best_metric_epoch = 0
    os.makedirs(root_dir, exist_ok=True)
    if metric > best_metric:
        best_metric = metric
        best_metric_epoch = epoch
        torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
        mlflow.pytorch.log_model(model, "model")
        print("Saved new best metric model")
    else:
        best_metric_epoch = best_metric_epoch
    return best_metric, best_metric_epoch

def entropy_volume(vol_input: torch.Tensor, dimension: int) -> torch.Tensor:
    """
    Calculate the entropy of a given volumetric input with repetitions and channels.
    
    Parameters:
        vol_input: A PyTorch tensor of probabilities with shape [MC samples, Channels, *Spatial dimensions].
        dimension: The spatial dimensionality of the volumetric data (2 or 3).
    
    Returns:
        A tensor representing the computed entropy volume, with the same spatial dimensions as the input volume.
    """
    vol_input = vol_input.clamp(min=1e-5)  # Ensure no log(0) issues
    t_avg = torch.mean(vol_input, dim=0)  # Average over MC samples
    t_log = torch.log(t_avg)
    t_entropy = -torch.sum(t_avg * t_log, dim=0)  # Sum over channels
    return t_entropy

def select_data_by_uncertainty_with_sw_inference(
    model: Module, 
    root_dir: str, 
    data_loader: DataLoader,
    device: torch.device,
    unlabelled_files: List[str],  
    n: int = 3, 
    mc_samples: int = 3, 
    roi_size: Tuple[int, int, int] = (160, 160, 160), 
    sw_batch_size: int = 4
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Perform Monte Carlo (MC) simulations to select samples from an unlabeled dataset based on uncertainty. 
    Uncertainty is estimated using the entropy of predictions obtained from multiple stochastic forward passes 
    (enabled by MC Dropout) through a segmentation model.

    Parameters:
        model: A PyTorch module representing the segmentation model equipped with MC Dropout.
        root_dir: The directory where the model's best checkpoint is stored.
        data_loader: DataLoader object providing access to the unlabeled dataset.
        device: The device (CPU or CUDA) where computations should be performed.
        unlabelled_files: A list containing the file paths of unlabeled data samples.
        n: The number of samples to select based on the highest uncertainty.
        mc_samples: The number of Monte Carlo forward passes for uncertainty estimation.
        roi_size: The size of the region of interest to process in one forward pass.
        sw_batch_size: The number of sliding windows to process in parallel during inference.

    Returns:
        A tuple containing:
        - indices of the selected samples based on uncertainty,
        - their corresponding uncertainties,
        - and the filenames of the selected samples.
    """
    model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth"), map_location=device))
    model.train()  # MC Dropout enabled
    
    uncertainties = []

    for data in data_loader:
        # Assuming inputs are on GPU if available
        inputs = data["image"].to(device)
        mc_predictions = torch.zeros((mc_samples, *inputs.shape[1:]), device=device)
        
        for mc_sample in range(mc_samples):
            with torch.no_grad():
                outputs = sliding_window_inference(inputs, roi_size, sw_batch_size, model, overlap=0.5)
                outputs = outputs[0].detach().cpu()[1, :, :, :]
                mc_predictions[mc_sample] = torch.softmax(outputs, dim=1)

        # Compute entropy across all MC samples for each voxel
        mc_entropy = entropy_volume(mc_predictions, 3 if len(inputs.shape) == 5 else 2)
        sample_uncertainty = torch.mean(mc_entropy).item()  # Mean entropy across spatial dimensions
        uncertainties.append(sample_uncertainty)

    indices = np.argsort(uncertainties) # Select n samples with highest uncertainty
    predicted_labels = [unlabelled_files[i] for i in indices]

    return indices, np.array(uncertainties)[indices], predicted_labels

def log_to_mlflow(indices: np.ndarray, uncertainties: np.ndarray, filenames: List[str]) -> None:
    """
    Logs selected data indices, their uncertainties, and filenames as MLflow artifacts.

    Parameters:
        indices: Indices of selected samples.
        uncertainties: Uncertainties associated with the selected samples.
        filenames: Filenames of the selected samples.
    """
    # Save indices, uncertainties, and filenames to files
    np.savetxt("indices.txt", indices, fmt='%i')
    np.savetxt("uncertainties.txt", uncertainties)
    with open("filenames.json", 'w') as f:
        json.dump(filenames, f)

    # Log files as artifacts
    mlflow.log_artifact("indices.txt", artifact_path=f'active_learning')
    mlflow.log_artifact("uncertainties.txt", artifact_path=f'active_learning')
    mlflow.log_artifact("filenames.json", artifact_path=f'active_learning')

    # Cleanup
    os.remove("indices.txt")
    os.remove("uncertainties.txt")
    os.remove("filenames.json")

def save_prediction_as_nifti(
    model: torch.nn.Module, 
    loader: torch.utils.data.DataLoader, 
    device: torch.device, 
    op_dir: str, 
    subdir: str, 
    root_dir: str,
    filenames: List[Dict[str, str]]
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
    post_pred = get_post_transforms_unlabelled()
    os.makedirs(op_dir, exist_ok=True)
    os.makedirs(f'{op_dir}/{subdir}', exist_ok=True)

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


def run_training(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, 
                 val_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, 
                 loss_function: torch.nn.Module, device: torch.device, max_epochs: int, 
                 val_interval: int, root_dir: str):
    """
    Run the training and validation loop.

    Parameters:
    - model: torch.nn.Module - The model to train.
    - train_loader: DataLoader - DataLoader for training data.
    - val_loader: DataLoader - DataLoader for validation data.
    - optimizer: torch.optim.Optimizer - Optimizer for the model parameters.
    - loss_function: torch.nn.Module - Loss function.
    - device: torch.device - Device to train on.
    - max_epochs: int - Number of epochs to train for.
    - val_interval: int - Interval between validations.
    - root_dir: str - Directory to save the best model.
    """
    best_metric = -1
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
    post_label = Compose([AsDiscrete(to_onehot=2)])
    dice_metric = DiceMetric(include_background=True, reduction="mean")

    for epoch in range(max_epochs):
        print(f"Epoch {epoch + 1}/{max_epochs}")
        epoch_loss = train_epoch(model, train_loader, optimizer, loss_function, device)
        print(f"Average training loss: {epoch_loss:.4f}")
        mlflow.log_metric("training_loss", epoch_loss, step=epoch)

        if (epoch + 1) % val_interval == 0:
            val_loss, metric = validate(model, val_loader, loss_function, device, post_pred, post_label, dice_metric)
            print(f"Validation loss: {val_loss:.4f}, Dice Metric: {metric:.4f}")
            mlflow.log_metric("validation_loss", val_loss, step=epoch)
            mlflow.log_metric("dice_metric", metric, step=epoch)
            best_metric, _ = save_best_model(metric, best_metric, model, epoch + 1, root_dir)

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



def prepare_training_environment(config: Dict) -> Tuple[torch.nn.Module, Dict[str, DataLoader], Optimizer, DiceLoss, torch.device]:
    """
    Prepare the model, data loaders, optimizer, and loss function for training.

    Parameters:
        config: A dictionary containing training environment configurations.

    Returns:
        A tuple containing the model, data loaders, optimizer, loss function, and device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    loaders = create_data_loaders(
        data_dir=config["data_loader_params"]["data_dir"],
        batch_size=config["data_loader_params"]["batch_size"],
        num_workers=config["data_loader_params"]["num_workers"]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["model_params"]["learning_rate"])
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    return model, loaders, optimizer, loss_function, device

def log_nifti_directory_as_artifacts(directory_path: str) -> None:
    """
    Logs all NIfTI files in a directory as artifacts to the MLflow server.

    Parameters:
        directory_path: The path to the directory containing NIfTI files to be logged.
    """
    mlflow.log_artifacts(directory_path, artifact_path="predictions")
    shutil.rmtree(directory_path)

def execute_training_and_logging(
    model: torch.nn.Module,
    loaders: Dict[str, DataLoader],
    optimizer: Optimizer,
    loss_function: DiceLoss,
    device: torch.device,
    config: Dict
) -> None:
    """
    Run training and save predictions as NIfTI files, then log to MLflow.

    Parameters:
        model: The neural network model to be trained.
        loaders: A dictionary of DataLoader instances for training, validation, and potentially testing.
        optimizer: The optimizer for training the model.
        loss_function: The loss function used during training.
        device: The device (CPU or CUDA) on which to perform training.
        config: A dictionary containing training and logging configuration.
    """
    # Placeholder for run_training function implementation
    run_training(model, loaders["train"], loaders["val"], optimizer, loss_function, device, max_epochs=config["training_params"]["max_epochs"], val_interval=config["training_params"]["val_interval"], root_dir=config["root_dir"])

    # Placeholder for save_prediction_as_nifti function implementation
    loaders_predictions = create_data_loaders_predictions(data_dir=config["data_loader_params"]["data_dir"], batch_size=1, num_workers=config["data_loader_params"]["num_workers"])
    save_prediction_as_nifti(model, loaders_predictions["val"],device, "./predictions","val", config["root_dir"], loaders_predictions["val_files"])
    save_prediction_as_nifti(model, loaders_predictions["test"],device, "./predictions","test", config["root_dir"], loaders_predictions["test_files"])
    save_prediction_as_nifti(model, loaders_predictions["train"],device, "./predictions","train", config["root_dir"], loaders_predictions["train_files"])
    save_prediction_as_nifti(model, loaders_predictions["unlabelled"],device, "./predictions","unlabelled", config["root_dir"], loaders_predictions["unlabelled_files"])

    # Log NIfTI directory as artifacts
    log_nifti_directory_as_artifacts("./predictions")


def get_latest_commit_hash(repo_path: str) -> str:
    """
    Retrieve the latest commit hash of a Git repository.

    Parameters:
        repo_path: A string representing the file system path to the Git repository.

    Returns:
        The latest commit hash as a string.
    """
    cmd: List[str] = ['git', '-C', repo_path, 'rev-parse', 'HEAD']
    commit_hash: str = subprocess.check_output(cmd).decode('utf-8').strip()
    return commit_hash


def log_data_version(repo_path: str) -> None:
    """
    Log the latest commit hash of a Git repository to MLflow as a tag.

    Parameters:
        repo_path: A string representing the file system path to the Git repository.
    """
    commit_hash: str = get_latest_commit_hash(repo_path)
    mlflow.set_tag('data_repo_commit_hash', commit_hash)
    

def main():
    
    config = load_config()
    setup_mlflow(config)
    log_data_version(f'{config["data_loader_params"]["data_dir"]}/Spleen-stratified')

    model, loaders, optimizer, loss_function, device = prepare_training_environment(config)
    execute_training_and_logging(model, loaders, optimizer, loss_function, device, config)
    indices,uncertainties,files = select_data_by_uncertainty_with_sw_inference(model,config["root_dir"], loaders["unlabelled"],device, loaders["unlabelled_files"] )
    log_to_mlflow(indices, uncertainties, files)

    mlflow.end_run()

if __name__ == "__main__":
    main()
