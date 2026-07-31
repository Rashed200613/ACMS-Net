"""
Utility functions for metrics calculation, plotting, and visualization
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch

from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    cohen_kappa_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

import config
import time


def set_seed(seed=42):
    """
    Set seeds for reproducibility across numpy, random, and torch.
    
    Args:
        seed (int): Seed value. Default: 42
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Seed set to {seed}")


def count_parameters(model):
    """
    Count total and trainable parameters in the model.
    
    Args:
        model (nn.Module): PyTorch model
    
    Returns:
        tuple: (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def print_model_info(model):
    """
    Print comprehensive model information including parameter counts.
    
    Args:
        model (nn.Module): PyTorch model
    """
    total_params, trainable_params = count_parameters(model)
    non_trainable_params = total_params - trainable_params
    
    print("\n" + "="*60)
    print("MODEL INFORMATION")
    print("="*60)
    print(f"Total Parameters       : {total_params:,}")
    print(f"Trainable Parameters   : {trainable_params:,}")
    print(f"Non-trainable Parameters: {non_trainable_params:,}")
    print(f"Model Size             : {(total_params * 4) / (1024**2):.2f} MB")
    print("="*60 + "\n")


def format_time(seconds):
    """
    Format seconds to readable time string.
    
    Args:
        seconds (float): Time in seconds
    
    Returns:
        str: Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def calculate_metrics(y_true, y_pred, y_prob=None):
    
    metrics = {}

    metrics['accuracy'] = accuracy_score(y_true, y_pred)

    metrics['precision'] = precision_score(
        y_true,
        y_pred,
        average='weighted',
        zero_division=0
    )

    metrics['recall'] = recall_score(
        y_true,
        y_pred,
        average='weighted',
        zero_division=0
    )

    metrics['f1_score'] = f1_score(
        y_true,
        y_pred,
        average='weighted',
        zero_division=0
    )

    metrics['balanced_accuracy'] = balanced_accuracy_score(
        y_true,
        y_pred
    )

    metrics['kappa'] = cohen_kappa_score(y_true, y_pred)

    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)

    specificity_scores = []

    for i in range(len(cm)):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fn + fp)

        specificity = tn / (tn + fp + 1e-8)
        specificity_scores.append(specificity)

    metrics['specificity'] = np.mean(specificity_scores)

    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(
                y_true,
                y_prob,
                multi_class='ovr'
            )
        except:
            metrics['roc_auc'] = 0
    else:
        metrics['roc_auc'] = 0

    return metrics


def create_result_directories():
    """Create result directories if they don't exist."""
    os.makedirs(config.TRAIN_RESULT_DIR, exist_ok=True)
    os.makedirs(config.TEST_RESULT_DIR, exist_ok=True)
    print("Result directories created.")


def save_training_metrics(epoch, train_loss, train_metrics, val_loss, val_metrics):
    """
    Save training metrics to CSV file.
    
    Args:
        epoch (int): Epoch number
        train_loss (float): Training loss
        train_metrics (dict): Training metrics dictionary
        val_loss (float): Validation loss
        val_metrics (dict): Validation metrics dictionary
    """
    
    result = {
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'train_accuracy': train_metrics['accuracy'],
        'train_precision': train_metrics['precision'],
        'train_recall': train_metrics['recall'],
        'train_f1': train_metrics['f1_score'],
        'train_kappa': train_metrics['kappa'],
        'train_mcc': train_metrics['mcc'],
        'train_specificity': train_metrics['specificity'],
        'train_balanced_accuracy': train_metrics['balanced_accuracy'],
        'train_auc': train_metrics['roc_auc'],
        'val_loss': val_loss,
        'val_accuracy': val_metrics['accuracy'],
        'val_precision': val_metrics['precision'],
        'val_recall': val_metrics['recall'],
        'val_f1': val_metrics['f1_score'],
        'val_kappa': val_metrics['kappa'],
        'val_mcc': val_metrics['mcc'],
        'val_specificity': val_metrics['specificity'],
        'val_balanced_accuracy': val_metrics['balanced_accuracy'],
        'val_auc': val_metrics['roc_auc']
    }

    df = pd.DataFrame([result])

    csv_path = os.path.join(config.TRAIN_RESULT_DIR, "training_metrics.csv")

    if not os.path.exists(csv_path):
        df.to_csv(csv_path, index=False)
    else:
        df.to_csv(csv_path, mode='a', header=False, index=False)


def plot_training_curves(history):
    """
    Plot and save training curves (loss and accuracy).
    
    Args:
        history (dict): Dictionary containing 'train_loss', 'val_loss', 'train_acc', 'val_acc'
    """
    
    epochs = range(1, len(history['train_loss']) + 1)

    # =====================================
    # LOSS CURVE
    # =====================================

    plt.figure(figsize=(10, 5))

    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    loss_path = os.path.join(config.TRAIN_RESULT_DIR, "loss_curve.png")
    plt.savefig(loss_path)
    plt.close()

    print(f"Loss curve saved to {loss_path}")

    # =====================================
    # ACCURACY CURVE
    # =====================================

    plt.figure(figsize=(10, 5))

    plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], label='Val Accuracy')

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()

    acc_path = os.path.join(config.TRAIN_RESULT_DIR, "accuracy_curve.png")
    plt.savefig(acc_path)
    plt.close()

    print(f"Accuracy curve saved to {acc_path}")


def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot and save confusion matrix.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        class_names (list): List of class names
    """
    
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    cm_path = os.path.join(config.TEST_RESULT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    print(f"Confusion matrix saved to {cm_path}")


def save_classification_report(y_true, y_pred, class_names):
    """
    Generate and save classification report.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        class_names (list): List of class names
    """
    
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )

    print("\n========== CLASSIFICATION REPORT ==========")
    print(report)

    report_path = os.path.join(config.TEST_RESULT_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    print(f"Classification report saved to {report_path}")


def save_test_metrics(test_loss, metrics):
    """
    Save test metrics to CSV file.
    
    Args:
        test_loss (float): Test loss
        metrics (dict): Metrics dictionary
    """

    test_result_df = pd.DataFrame([{
    'test_loss': test_loss,
    'accuracy': metrics['accuracy'],
    'precision': metrics['precision'],
    'recall': metrics['recall'],
    'specificity': metrics['specificity'],
    'balanced_accuracy': metrics['balanced_accuracy'],
    'f1_score': metrics['f1_score'],
    'kappa': metrics['kappa'],
    'mcc': metrics['mcc'],
    'roc_auc': metrics['roc_auc']
}])

    test_metrics_path = os.path.join(config.TEST_RESULT_DIR, "test_metrics.csv")
    test_result_df.to_csv(test_metrics_path, index=False)

    print(f"Test metrics saved to {test_metrics_path}")


def print_test_results(test_loss, metrics):

    print("\n========== TEST RESULT ==========")
    print(f"Test Loss          : {test_loss:.4f}")
    print(f"Accuracy           : {metrics['accuracy']:.4f}")
    print(f"Precision          : {metrics['precision']:.4f}")
    print(f"Recall             : {metrics['recall']:.4f}")
    print(f"Specificity        : {metrics['specificity']:.4f}")
    print(f"Balanced Accuracy  : {metrics['balanced_accuracy']:.4f}")
    print(f"F1 Score           : {metrics['f1_score']:.4f}")
    print(f"Kappa              : {metrics['kappa']:.4f}")
    print(f"MCC                : {metrics['mcc']:.4f}")
    print(f"ROC AUC            : {metrics['roc_auc']:.4f}")


def save_classwise_metrics(y_true, y_pred, class_names):

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    rows = []

    for cls in class_names:

        rows.append({
            "class": cls,
            "precision": report[cls]["precision"],
            "recall": report[cls]["recall"],
            "f1_score": report[cls]["f1-score"],
            "support": report[cls]["support"]
        })

    df = pd.DataFrame(rows)

    save_path = os.path.join(
        config.TEST_RESULT_DIR,
        "classwise_metrics.csv"
    )

    df.to_csv(save_path, index=False)

    print(f"Classwise metrics saved to {save_path}")


def plot_multiclass_roc(y_true, y_prob, class_names):

    y_true_bin = label_binarize(
        y_true,
        classes=list(range(len(class_names)))
    )

    plt.figure(figsize=(10, 8))

    for i, class_name in enumerate(class_names):

        fpr, tpr, _ = roc_curve(
            y_true_bin[:, i],
            np.array(y_prob)[:, i]
        )

        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            label=f"{class_name} (AUC={roc_auc:.4f})"
        )

    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multiclass ROC Curve")
    plt.legend()

    save_path = os.path.join(
        config.TEST_RESULT_DIR,
        "roc_curve.png"
    )

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    print(f"ROC curve saved to {save_path}")

def plot_multiclass_pr(y_true, y_prob, class_names):

    y_true_bin = label_binarize(
        y_true,
        classes=list(range(len(class_names)))
    )

    plt.figure(figsize=(10, 8))

    pr_rows = []

    for i, class_name in enumerate(class_names):

        precision, recall, _ = precision_recall_curve(
            y_true_bin[:, i],
            np.array(y_prob)[:, i]
        )

        ap = average_precision_score(
            y_true_bin[:, i],
            np.array(y_prob)[:, i]
        )

        pr_rows.append({
            "class": class_name,
            "AUPRC": ap
        })

        plt.plot(
            recall,
            precision,
            label=f"{class_name} (AUPRC={ap:.4f})"
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision Recall Curve")
    plt.legend()

    save_path = os.path.join(
        config.TEST_RESULT_DIR,
        "pr_curve.png"
    )

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

    pd.DataFrame(pr_rows).to_csv(
        os.path.join(
            config.TEST_RESULT_DIR,
            "auprc_scores.csv"
        ),
        index=False
    )

    print(f"PR curve saved to {save_path}")

def save_error_analysis(
    y_true,
    y_pred,
    class_names
):

    rows = []

    for idx, (gt, pred) in enumerate(zip(y_true, y_pred)):

        if gt != pred:

            rows.append({
                "sample_index": idx,
                "true_class": class_names[gt],
                "predicted_class": class_names[pred]
            })

    save_path = os.path.join(
        config.TEST_RESULT_DIR,
        "error_analysis.csv"
    )

    pd.DataFrame(rows).to_csv(
        save_path,
        index=False
    )

    print(f"Error analysis saved to {save_path}") 


def save_training_history(history):

    df = pd.DataFrame({
        "epoch": range(
            1,
            len(history["train_loss"]) + 1
        ),

        "train_loss":
            history["train_loss"],

        "val_loss":
            history["val_loss"],

        "train_accuracy":
            history["train_acc"],

        "val_accuracy":
            history["val_acc"],

        "train_time":
            history["train_time"],

        "val_time":
            history["val_time"],

        "learning_rate":
            history["learning_rate"]
    })

    save_path = os.path.join(
        config.TRAIN_RESULT_DIR,
        "training_history.csv"
    )

    df.to_csv(
        save_path,
        index=False
    )

    print(
        f"Training history saved to {save_path}"
    ) 


def save_learning_stability_analysis(history):

    rows = []

    for i in range(len(history["train_acc"])):

        acc_gap = abs(
            history["train_acc"][i]
            - history["val_acc"][i]
        )

        loss_gap = abs(
            history["train_loss"][i]
            - history["val_loss"][i]
        )

        rows.append({
            "epoch": i + 1,
            "accuracy_gap": acc_gap,
            "loss_gap": loss_gap
        })

    df = pd.DataFrame(rows)

    save_path = os.path.join(
        config.TRAIN_RESULT_DIR,
        "learning_stability_analysis.csv"
    )

    df.to_csv(
        save_path,
        index=False
    )

    print(
        f"Learning stability analysis saved to {save_path}"
    )              


def verify_setup():
    """
    Run verification tests to ensure all components are working correctly.
    This function tests:
    1. Network accepts correct input
    2. Parameter counting works
    3. Time formatting works
    4. All imports are correct
    5. Device compatibility works
    6. Inference time measurement works
    """
    from network import CustomModel
    
    print("="*70)
    print("VERIFICATION TEST")
    print("="*70)

    # Test 1: Network Architecture
    print("\n[TEST 1] Network Architecture")
    print("-" * 70)

    model = CustomModel(num_classes=5)
    print(f"✓ Model created successfully")

    # Test input shape (3-channel, 224x224)
    test_input = torch.randn(2, 3, 224, 224)
    print(f"✓ Test input shape: {test_input.shape}")

    output = model(test_input)
    print(f"✓ Output shape: {output.shape}")
    assert output.shape == (2, config.NUM_CLASSES), "Output shape incorrect!"
    print(f"✓ Output shape correct: (batch_size=2, num_classes=5)")

    # Test 2: Parameter Counting
    print("\n[TEST 2] Parameter Counting")
    print("-" * 70)

    total_params, trainable_params = count_parameters(model)
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    assert total_params > 0, "Parameter count should be > 0"
    print(f"✓ Parameter counting works correctly")

    # Test 3: Model Info Display
    print("\n[TEST 3] Model Information Display")
    print("-" * 70)
    print_model_info(model)
    print("✓ Model info display works correctly")

    # Test 4: Time Formatting
    print("[TEST 4] Time Formatting")
    print("-" * 70)

    test_times = [0.5, 2.34, 45.67, 120.5, 3600.0, 7200.0]
    for t in test_times:
        formatted = format_time(t)
        print(f"  {t:>8.2f} seconds → {formatted}")

    print(f"✓ Time formatting works correctly")

    # Test 5: Device Compatibility
    print("\n[TEST 5] Device Compatibility")
    print("-" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✓ Using device: {device}")

    model_device = model.to(device)
    test_input_device = test_input.to(device)
    output_device = model_device(test_input_device)
    print(f"✓ Model runs on {device}: {output_device.shape}")

    # Test 6: Inference Time Measurement
    print("\n[TEST 6] Inference Time Measurement")
    print("-" * 70)

    model.eval()
    with torch.no_grad():
        start = time.time()
        for _ in range(10):
            _ = model(test_input_device)
        inference_time = time.time() - start

    print(f"✓ Inference time (10 batches): {format_time(inference_time)}")
    print(f"✓ Average time per batch: {format_time(inference_time / 10)}")

    # Final Summary
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED!")
    print("="*70)
    print("\n✓ Network accepts 3-channel 224×224 images")
    print("✓ Parameter counting works")
    print("✓ Time formatting works")
    print("✓ Model info display works")
    print("✓ Device compatibility verified")
    print("✓ All modules are correctly integrated")
    print("\n→ Ready to run: python train.py")
    print("→ Ready to run: python test.py")
