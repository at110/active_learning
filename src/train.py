import os
import torch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.transforms import Compose, AsDiscrete
from monai.data import decollate_batch
from typing import Tuple
from model import build_model  # Adjust the import path as necessary
from data_loader import create_data_loaders  # Adjust the import path as necessary
import mlflow
import json
import numpy as np
from scipy.stats import entropy
from typing import List, Tuple
import nibabel as nib
from monai.handlers.utils import from_engine


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

def select_data_by_uncertainty_with_sw_inference(model, root_dir, data_loader,device: torch.device,unlabelled_files,  n=3, mc_samples=3, roi_size=(160, 160, 160), sw_batch_size=4):
    """
    Selects n samples from the unlabeled dataset based on uncertainty using MC Dropout and sliding window inference.

    Args:
        model: The segmentation model with MC Dropout enabled.
        unlabeled_dataset: The dataset of unlabeled samples.
        n: Number of samples to select based on highest uncertainty.
        mc_samples: Number of Monte Carlo forward passes for uncertainty estimation.
        roi_size: Size of the ROI to process in one forward pass.
        sw_batch_size: Number of sliding windows to process in parallel.

    Returns:
        indices: Indices of the selected samples based on uncertainty.
    """
    # Ensure model is in training mode to use MC Dropout during inference
    model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth"), map_location=device))

    model.train()
    uncertainties = []

    for i, data in enumerate(data_loader):
        #inputs = inputs.cuda()  # Assuming use of GPU
        mc_entropies = []

        for _ in range(mc_samples):
            with torch.no_grad():
                # Use sliding_window_inference for each MC sample
                outputs = sliding_window_inference(data["image"].to(device), roi_size, sw_batch_size, model, overlap=0.5)
                # Convert softmax probabilities
                probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
                # Calculate pixel-wise entropy for the current MC sample
                mc_entropy = entropy(probabilities, base=2, axis=1)
                mc_entropies.append(mc_entropy)
        
        # Average entropy across MC samples to get a single uncertainty value per voxel
        mean_entropy = np.mean(mc_entropies, axis=0)
        # Average entropy across all voxels for a sample to get overall uncertainty
        sample_uncertainty = np.mean(mean_entropy)
        
        uncertainties.append(sample_uncertainty)
    # Select n samples with the highest uncertainty
    indices = np.argsort(uncertainties)
    print(indices)

    predicted_labels = [unlabelled_files[ indices[index]]['image'] for index in indices]
    return indices, np.array(uncertainties)[indices],predicted_labels

def log_to_mlflow(indices: np.ndarray, uncertainties: np.ndarray, filenames: List[str]):
    # Save indices, uncertainties, and filenames to files
    np.savetxt("indices.txt", indices, fmt='%i')
    np.savetxt("uncertainties.txt", uncertainties)
    with open("filenames.json", 'w') as f:
        json.dump(filenames, f)

    # Log files as artifacts
    mlflow.log_artifact("indices.txt")
    mlflow.log_artifact("uncertainties.txt")
    mlflow.log_artifact("filenames.json")

    # Cleanup
    os.remove("indices.txt")
    os.remove("uncertainties.txt")
    os.remove("filenames.json")

def save_prediction_as_nifti(model, loader, index, device, op_dir, root_dir, output_filename,post_pred: Compose):
    """
    Load the model's best state, predict labels for the second batch in the validation loader,
    and save these predictions as a NIfTI file with an explicit data type conversion to int16.

    Parameters:
    - model: The trained PyTorch model.
    - val_loader: DataLoader for the validation dataset.
    - device: The device (e.g., "cpu" or "cuda") on which the model is running.
    - root_dir: Directory where the best model checkpoint is saved.
    - output_filename: Name of the output NIfTI file to save the predictions.
    """
    # Load the best saved model state
    post_label = Compose([AsDiscrete(to_onehot=2)])
    os.makedirs(op_dir, exist_ok=True)
    model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model.pth"), map_location=device))
    model.eval()
    model.to(device)
    subject_num = 0 
    with torch.no_grad():
        for data in loader:
            if subject_num == index:  # Proceed only for the second batch
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
                nib.save(pred_nifti, os.path.join(op_dir, output_filename))
                print(f"Saved predictions as NIfTI file at: {os.path.join(op_dir, output_filename)}")
                break  # Exit the loop after processing the required batch
            subject_num+=1

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

def main():
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    
    mlflow.set_tracking_uri(config["mlflow_tracking_uri"])
    mlflow.set_experiment(config["experiment_name"])  # Set your experiment name
    mlflow.start_run(log_system_metrics=True)
   

    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
    mlflow.log_param("learning_rate", config["model_params"]["learning_rate"])
    mlflow.log_param("batch_size", config["model_params"]["batch_size"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    loaders = create_data_loaders(data_dir=".", batch_size=2, num_workers=4)
    optimizer = torch.optim.Adam(model.parameters())
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    run_training(model, loaders["train"], loaders["val"], optimizer, loss_function, device, max_epochs=config["training_params"]["max_epochs"], val_interval=config["training_params"]["val_interval"], root_dir=config["root_dir"])
    indices,uncertainties,files = select_data_by_uncertainty_with_sw_inference(model,config["root_dir"], loaders["unlabelled"],device, loaders["unlabelled_files"] )
    log_to_mlflow(indices, uncertainties, files)
    save_prediction_as_nifti(model, loaders["val"],device, "./output", config["root_dir"], loaders["val_files"])
    mlflow.end_run()
if __name__ == "__main__":
    main()
