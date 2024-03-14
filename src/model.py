import torch
from monai.networks.nets import UNet
from monai.networks.layers import Norm

def build_model() -> torch.nn.Module:
    """
    Build and return the UNet model configured for spleen segmentation.

    Returns:
    - torch.nn.Module: The UNet model.
    """
    model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    dropout= 0.3,
    num_res_units=2,
    norm=Norm.BATCH,
    )

    return model