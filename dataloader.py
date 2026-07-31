import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import config


def get_transforms():
    """
    Returns data transformation pipelines for training and validation/test sets.
    
    Returns:
        tuple: (train_transform, val_test_transform)
    """
    
    train_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225])
    ])

    return train_transform, val_test_transform


def get_dataloaders():
    """
    Creates and returns train, validation, and test dataloaders.
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, class_names)
    """
    
    train_transform, val_test_transform = get_transforms()

    # =====================================
    # LOAD DATASETS
    # =====================================

    train_dataset = datasets.ImageFolder(
        config.TRAIN_DIR,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        config.VAL_DIR,
        transform=val_test_transform
    )

    test_dataset = datasets.ImageFolder(
        config.TEST_DIR,
        transform=val_test_transform
    )

    # =====================================
    # CREATE DATALOADERS
    # =====================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS
    )

    class_names = train_dataset.classes

    return train_loader, val_loader, test_loader, class_names
