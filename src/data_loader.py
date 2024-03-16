import os
import glob
from typing import Dict, List, Tuple
import numpy as np
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

def get_train_transforms() -> Compose:
    """Defines and returns the transformation sequence for training data."""
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=200, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(0.79, 0.79, 2.5), mode=("bilinear", "nearest")),
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=(64, 64, 64),
                               pos=1, neg=1, num_samples=4, image_key="image", image_threshold=0),
        RandAffined(keys=['image', 'label'], mode=('bilinear', 'nearest'), prob=1.0, spatial_size=(64, 64, 64),
                    rotate_range=(0, 0, np.pi/15), scale_range=(0.1, 0.1, 0.1)),
    ])

def get_val_transforms() -> Compose:
    """Defines and returns the transformation sequence for validation data."""
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=200, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(0.79, 0.79, 2.5), mode=("bilinear", "nearest")),
    ])

def get_unlabelled_transforms() -> Compose:
    """Defines and returns the transformation sequence for unlabelled data."""
    return Compose([
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(0.79, 0.79, 2.5), mode="bilinear"),
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=200, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image"], source_key="image"),
    ])

def get_post_transforms_unlabelled() -> Compose:
    """Defines and returns post-processing transformation sequence for unlabelled data predictions."""
    return Compose([
        Invertd(keys="pred", transform=get_unlabelled_transforms(), orig_keys="image",
                meta_keys="pred_meta_dict", orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict", nearest_interp=False, to_tensor=True, device="cpu"),
        AsDiscreted(keys="pred", argmax=True, to_onehot=2),
    ])

def create_data_loaders(data_dir: str, batch_size: int = 2, num_workers: int = 4) -> Dict[str, DataLoader]:
    """
    Creates and returns DataLoaders for training, validation, and unlabelled datasets.

    Parameters:
    - data_dir: str - Directory where the datasets are located.
    - batch_size: int - Batch size for training and validation loaders. Default is 2.
    - num_workers: int - Number of worker processes for DataLoader. Default is 4.

    Returns:
    - Dict[str, DataLoader]: A dictionary with 'train', 'val', and 'unlabelled' DataLoaders.
    """
    train_images = sorted(glob.glob(os.path.join(data_dir, "Spleen-stratified/imagesTrain", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "Spleen-stratified/labelsTrain", "*.nii.gz")))
    train_files = [{"image": img, "label": lbl} for img, lbl in zip(train_images, train_labels)]

    val_images = sorted(glob.glob(os.path.join(data_dir, "Spleen-stratified/imagesVal", "*.nii.gz")))
    val_labels = sorted(glob.glob(os.path.join(data_dir, "Spleen-stratified/labelsVal", "*.nii.gz")))
    val_files = [{"image": img, "label": lbl} for img, lbl in zip(val_images, val_labels)]

    test_images = sorted(glob.glob(os.path.join(data_dir, "Spleen-stratified/imagesTest", "*.nii.gz")))
    test_labels = sorted(glob.glob(os.path.join(data_dir, "Spleen-stratified/labelsTest", "*.nii.gz")))
    test_files = [{"image": img, "label": lbl} for img, lbl in zip(val_images, val_labels)]

    unlabelled_images = sorted(glob.glob(os.path.join(data_dir, "Spleen-stratified/imagesUnlabelled", "*.nii.gz")))
    unlabelled_files = [{"image": img} for img in unlabelled_images]

    train_ds = CacheDataset(data=train_files, transform=get_train_transforms(), cache_rate=1.0, num_workers=num_workers)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_ds = CacheDataset(data=val_files, transform=get_val_transforms(), cache_rate=1.0, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers)

    test_ds = CacheDataset(data=test_files, transform=get_val_transforms(), cache_rate=1.0, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=num_workers)

    unlabelled_ds = CacheDataset(data=unlabelled_files, transform=get_unlabelled_transforms(), cache_rate=1.0, num_workers=num_workers)
    unlabelled_loader = DataLoader(unlabelled_ds, batch_size=1, shuffle=False, num_workers=num_workers)

    return {"train": train_loader, "val": val_loader, "test":test_loader, "unlabelled": unlabelled_loader, "train_files":train_files, "val_files":val_files, "test_files": test_files, "unlabelled_files":unlabelled_files}
