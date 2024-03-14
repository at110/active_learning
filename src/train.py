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
        #torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model.pth"))
        mlflow.pytorch.log_model(model, "model")
        print("Saved new best metric model")
    else:
        best_metric_epoch = best_metric_epoch
    return best_metric, best_metric_epoch

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
    
    mlflow.set_tracking_uri("http://ec2-34-227-229-249.compute-1.amazonaws.com:5000/")
    mlflow.set_experiment("Default")  # Set your experiment name
    mlflow.start_run()
   
    mlflow.log_param("learning_rate", 0)
    mlflow.log_param("batch_size", 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    loaders = create_data_loaders(data_dir="./tests/", batch_size=2, num_workers=4)
    optimizer = torch.optim.Adam(model.parameters())
    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    run_training(model, loaders["train"], loaders["val"], optimizer, loss_function, device, max_epochs=6, val_interval=1, root_dir="./models")
    mlflow.end_run()
if __name__ == "__main__":
    main()
