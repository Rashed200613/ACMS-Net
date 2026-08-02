
"""
5-Fold Stratified Cross-Validation for ACMS-Net (Gallbladder Classification)

Ei script ki kore:
    1. Original dataset er train+val+test - shobgulo combine kore ekta
       single pool banay.
    2. sklearn StratifiedKFold (K=5) diye shei pool ke 5-ta fold e bhag
       kore (protik fold e class-ratio prai same thake).
    3. Protyek fold er jonno network.py theke ekta fresh CustomModel
       train kore (kono weight-sharing/leakage nai - protyek fold
       independent train hoy), tারপর shei fold-er held-out validation
       part e evaluate kore.
    4. 5-ta fold theke পাওয়া metrics (accuracy, precision, recall,
       f1, balanced accuracy, kappa, mcc, specificity, roc_auc)
       diye:
           - per_fold_metrics.csv        (fold-wise raw result)
           - statistics_report.csv       (Mean ± Std, publication table)
           - confidence_interval.csv     (95% CI, publication table)
       generate kore.

Usage:
    python kfold_train.py 
"""

import os
import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import config
from dataset import collect_all_samples, GallbladderFullDataset, get_transforms
from network import CustomModel
from utils import (
    set_seed,
    create_result_directories,
    calculate_metrics,
    count_parameters,
    print_model_info,
    format_time,
)
from statistical_analysis import save_full_statistics


# ==========================================================
# ONE EPOCH - TRAIN
# ==========================================================

def train_epoch(model, loader, criterion, optimizer, device):

    model.train()
    running_loss = 0.0

    for images, labels in tqdm(loader, desc="Training", leave=False):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


# ==========================================================
# ONE EPOCH - VALIDATE
# ==========================================================

def validate_epoch(model, loader, criterion, device):

    model.eval()
    running_loss = 0.0

    all_true, all_pred, all_prob = [], [], []

    with torch.no_grad():

        for images, labels in tqdm(loader, desc="Validating", leave=False):

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_true.extend(labels.cpu().numpy())
            all_pred.extend(preds.cpu().numpy())
            all_prob.extend(probs.cpu().numpy())

    val_loss = running_loss / len(loader)
    metrics = calculate_metrics(all_true, all_pred, all_prob)

    return val_loss, metrics, all_true, all_pred


# ==========================================================
# TRAIN ONE FOLD (fresh model, from scratch)
# ==========================================================

def run_fold(fold_idx, train_idx, val_idx, samples, class_names):

    print("\n" + "=" * 70)
    print(f"FOLD {fold_idx}/{config.K_FOLDS}")
    print("=" * 70)

    set_seed(config.SEED)   # protyek fold e same initialization condition

    train_transform, val_transform = get_transforms()

    train_dataset_full = GallbladderFullDataset(samples, train_transform)
    val_dataset_full = GallbladderFullDataset(samples, val_transform)

    train_subset = Subset(train_dataset_full, train_idx)
    val_subset = Subset(val_dataset_full, val_idx)

    train_loader = DataLoader(
        train_subset, batch_size=config.BATCH_SIZE,
        shuffle=True, num_workers=config.NUM_WORKERS
    )
    val_loader = DataLoader(
        val_subset, batch_size=config.BATCH_SIZE,
        shuffle=False, num_workers=config.NUM_WORKERS
    )

    model = CustomModel(num_classes=config.NUM_CLASSES).to(config.DEVICE)

    if fold_idx == 1:
        print_model_info(model)

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

    best_val_acc = 0.0
    best_state = None
    best_true, best_pred = None, None

    fold_start = time.time()

    for epoch in range(config.EPOCHS):

        train_loss = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_metrics, val_true, val_pred = validate_epoch(
            model, val_loader, criterion, config.DEVICE
        )

        scheduler.step(val_loss)

        print(
            f"[Fold {fold_idx}] Epoch [{epoch + 1}/{config.EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_metrics['accuracy']:.4f}"
        )

        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_state = copy.deepcopy(model.state_dict())
            best_true, best_pred = val_true, val_pred

    fold_time = time.time() - fold_start
    print(f"Fold {fold_idx} completed in {format_time(fold_time)} | Best Val Acc: {best_val_acc:.4f}")

    # ---- Final evaluation of this fold using its best checkpoint ----
    model.load_state_dict(best_state)
    _, final_metrics, final_true, final_pred = validate_epoch(
        model, val_loader, criterion, config.DEVICE
    )

    # Save this fold's best model checkpoint (optional, useful for reproducibility)
    fold_model_path = os.path.join(config.RESULT_DIR, f"fold_{fold_idx}_best_model.pth")
    torch.save(best_state, fold_model_path)

    final_metrics["fold"] = fold_idx
    final_metrics["fold_time_sec"] = fold_time

    return final_metrics


# ==========================================================
# MAIN
# ==========================================================

def main():

    set_seed(config.SEED)
    create_result_directories()

    print(f"Device: {config.DEVICE}")
    print(f"K-Folds: {config.K_FOLDS}")

    samples, class_names = collect_all_samples()
    labels = [s[1] for s in samples]

    print(f"Total combined samples (train+val+test): {len(samples)}")
    print(f"Classes ({len(class_names)}): {class_names}")

    skf = StratifiedKFold(
        n_splits=config.K_FOLDS,
        shuffle=True,
        random_state=config.SEED
    )

    fold_results = []

    metric_names = [
        "accuracy", "precision", "recall", "f1_score",
        "balanced_accuracy", "kappa", "mcc", "specificity", "roc_auc"
    ]
    ordered_cols = ["fold"] + metric_names + ["fold_time_sec"]

    per_fold_path = os.path.join(config.RESULT_DIR, "per_fold_metrics.csv")

    for fold_idx, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(labels)), labels), start=1
    ):
        fold_metrics = run_fold(fold_idx, train_idx, val_idx, samples, class_names)
        fold_results.append(fold_metrics)

        # ==========================================================
        # SAVE RESULTS COMPLETED SO FAR (ei fold shesh hobar por-e,
        # ekhon porjonto shob fold-er result csv-te update hoy)
        # ==========================================================

        results_df_so_far = pd.DataFrame(fold_results)
        results_df_so_far = results_df_so_far[ordered_cols]

        display_df_so_far = results_df_so_far.copy()
        for m in metric_names:
            display_df_so_far[m] = (display_df_so_far[m] * 100).round(2)

        display_df_so_far.to_csv(per_fold_path, index=False, encoding="utf-8-sig")

        print(f"\n[Fold {fold_idx}/{config.K_FOLDS}] Results so far saved to: {per_fold_path}")
        print(display_df_so_far.to_string(index=False))

    # ==========================================================
    # SHOB FOLD SHESH - FINAL COMBINED SAVE (Mean +/- SD row soho)
    # ==========================================================

    results_df = pd.DataFrame(fold_results)
    results_df = results_df[ordered_cols]

    # Raw (0-1 scale) values needed later for confidence_interval.csv
    metrics_dict = {m: results_df[m].tolist() for m in metric_names}

    # ---- Build a display copy in percentage (0-100), matching the report table ----
    display_df = results_df.copy()
    for m in metric_names:
        display_df[m] = (display_df[m] * 100).round(2)

    # ---- Mean +/- SD summary row (percentage, e.g. "96.78 ± 0.28") ----
    summary_row = {"fold": "Mean ± SD", "fold_time_sec": ""}
    for m in metric_names:
        vals = results_df[m].to_numpy() * 100
        mean = vals.mean()
        std = vals.std(ddof=1)
        summary_row[m] = f"{mean:.2f} \u00b1 {std:.2f}"

    final_df = pd.concat(
        [display_df, pd.DataFrame([summary_row])], ignore_index=True
    )
    final_df = final_df[ordered_cols]

    final_df.to_csv(per_fold_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("PER-FOLD RESULTS")
    print("=" * 70)
    print(final_df.to_string(index=False))
    print(f"\nSaved to: {per_fold_path}")

    # ==========================================================
    # 95% CONFIDENCE INTERVAL (separate file, not fold-wise)
    # ==========================================================

    save_full_statistics(metrics_dict)

    print("\nK-Fold Cross-Validation + Statistical Validation complete!")
    print(f"All results saved in: {config.RESULT_DIR}/")


if __name__ == "__main__":
    main()
