"""
Training script for Gallbladder Classification Model
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time

from tqdm import tqdm

import config
from dataset import get_dataloaders

# Loads CustomModel from whichever file config.NETWORK_FILE points to
# (network.py for the full model, abliation1.py for the ablation variant,
# etc.) so this script never needs to be edited when switching variants.
import importlib.util as _ilu


def _load_custom_model_class(network_file):
    spec = _ilu.spec_from_file_location("active_network_module", network_file)
    module = _ilu.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "CustomModel"):
        raise AttributeError(f"{network_file} does not define CustomModel.")
    return module.CustomModel


CustomModel = _load_custom_model_class(config.NETWORK_FILE)
print(f"[INFO] Using architecture: {config.NETWORK_FILE}  (variant: {config.MODEL_VARIANT})")

from utils import (
    set_seed,
    create_result_directories,
    calculate_metrics,
    save_training_metrics,
    plot_training_curves,
    save_training_history,
    save_learning_stability_analysis,
    print_test_results,
    count_parameters,
    print_model_info,
    format_time
)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch.
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        criterion (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        device (str): Device to train on ('cuda' or 'cpu')
    
    Returns:
        tuple: (train_loss, train_metrics, epoch_time)
    """
    
    model.train()
    
    epoch_start = time.time()

    train_loss = 0
    train_true = []
    train_pred = []
    train_prob = []

    for images, labels in tqdm(train_loader, desc="Training"):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)

        train_true.extend(labels.cpu().numpy())
        train_pred.extend(preds.cpu().numpy())
        train_prob.extend(probs.detach().cpu().numpy())

    train_loss /= len(train_loader)
    train_metrics = calculate_metrics(train_true, train_pred, train_prob)
    
    epoch_time = time.time() - epoch_start

    return train_loss, train_metrics, epoch_time


def validate_epoch(model, val_loader, criterion, device):
    """
    Validate for one epoch.
    
    Args:
        model (nn.Module): Model to validate
        val_loader (DataLoader): Validation data loader
        criterion (nn.Module): Loss function
        device (str): Device to validate on ('cuda' or 'cpu')
    
    Returns:
        tuple: (val_loss, val_metrics, epoch_time)
    """
    
    model.eval()
    
    epoch_start = time.time()

    val_loss = 0
    val_true = []
    val_pred = []
    val_prob = []

    with torch.no_grad():

        for images, labels in tqdm(val_loader, desc="Validating"):

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            val_true.extend(labels.cpu().numpy())
            val_pred.extend(preds.cpu().numpy())
            val_prob.extend(probs.cpu().numpy())

    val_loss /= len(val_loader)
    val_metrics = calculate_metrics(val_true, val_pred, val_prob)
    
    epoch_time = time.time() - epoch_start

    return val_loss, val_metrics, epoch_time


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs, model_save_path):
    """
    Main training loop.
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        criterion (nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler
        device (str): Device to train on
        epochs (int): Number of epochs
        model_save_path (str): Path to save best model
    
    Returns:
        dict: Training history with timing information
    """
    
    history = {
    'train_loss': [],
    'val_loss': [],
    'train_acc': [],
    'val_acc': [],
    'train_time': [],
    'val_time': [],
    'learning_rate': []
}

    best_val_acc = 0

    for epoch in range(epochs):

        print(f"\n{'='*70}")
        print(f"Epoch [{epoch+1}/{epochs}]")
        print(f"{'='*70}")

        # =====================================
        # TRAIN
        # =====================================

        train_loss, train_metrics, train_time = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        # =====================================
        # VALIDATION
        # =====================================

        val_loss, val_metrics, val_time = validate_epoch(
            model,
            val_loader,
            criterion,
            device
        )

        scheduler.step(val_loss)

        # =====================================
        # HISTORY
        # =====================================

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['train_time'].append(train_time)
        history['val_time'].append(val_time)
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        # =====================================
        # SAVE METRICS
        # =====================================

        save_training_metrics(epoch, train_loss, train_metrics, val_loss, val_metrics)

        # =====================================
        # SAVE BEST MODEL
        # =====================================

        if val_metrics['accuracy'] > best_val_acc:

            best_val_acc = val_metrics['accuracy']

            torch.save(model.state_dict(), model_save_path)

            print(f"✓ Best model saved! Val Acc: {val_metrics['accuracy']:.4f}")

        # =====================================
        # PRINT RESULTS
        # =====================================

        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val Loss  : {val_loss:.4f} | Val Acc  : {val_metrics['accuracy']:.4f}")
        print(f"\nTrain Time: {format_time(train_time)} | Val Time: {format_time(val_time)}")
        print(f"Epoch Time: {format_time(train_time + val_time)}")
        print(
            f"Learning Rate: "
            f"{optimizer.param_groups[0]['lr']:.8f}"
        )

    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)

    save_training_history(history)

    return history


def main():
    """Main training function."""
    
    # =====================================
    # SETUP
    # =====================================
    
    set_seed(config.SEED)
    create_result_directories()

    print(f"Device: {config.DEVICE}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Learning Rate: {config.LEARNING_RATE}")

    # =====================================
    # LOAD DATA
    # =====================================

    print("\nLoading datasets...")
    train_loader, val_loader, test_loader, class_names = get_dataloaders()

    print(f"Classes: {class_names}")
    print(f"Number of classes: {len(class_names)}")

    # =====================================
    # MODEL
    # =====================================

    print("\nInitializing model...")
    model = CustomModel(num_classes=config.NUM_CLASSES)
    model = model.to(config.DEVICE)
    
    # Display model information
    print_model_info(model)

    # =====================================
    # LOSS + OPTIMIZER + SCHEDULER
    # =====================================

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=config.SCHEDULER_MODE,
        patience=config.SCHEDULER_PATIENCE,
        factor=config.SCHEDULER_FACTOR
    )

    print("Loss, optimizer, scheduler ready.")

    # =====================================
    # TRAIN
    # =====================================

    history = train(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        config.DEVICE,
        config.EPOCHS,
        config.MODEL_SAVE_PATH
    )

    # =====================================
    # PLOT CURVES
    # =====================================

    print("\nGenerating training curves...")

    plot_training_curves(history)

    save_training_history(history)

    save_learning_stability_analysis(
        history
    )

    print("\nTraining pipeline complete!")
    print(f"Results saved to: {config.TRAIN_RESULT_DIR}/")


if __name__ == "__main__":
    main()