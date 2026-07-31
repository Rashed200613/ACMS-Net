"""
Testing/Evaluation script for Gallbladder Classification Model
"""

import os
from pyexpat import model

import torch
import torch.nn as nn
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
    print_test_results,
    save_test_metrics,
    plot_confusion_matrix,
    save_classification_report,
    save_classwise_metrics,
    print_model_info,
    plot_multiclass_roc,
    plot_multiclass_pr,
    save_error_analysis,
    format_time
)

from complexity import save_complexity_report

from explainability import (
    generate_gradcam,
    save_feature_maps,
    save_attention_maps,
    generate_tsne
)

from robustness import (
    robustness_report,
    save_external_dataset_result,
    save_attention_analysis
)

def evaluate(model, test_loader, criterion, device, class_names):
    """
    Evaluate model on test set.
    
    Args:
        model (nn.Module): Model to evaluate
        test_loader (DataLoader): Test data loader
        criterion (nn.Module): Loss function
        device (str): Device to evaluate on ('cuda' or 'cpu')
        class_names (list): List of class names
    
    Returns:
        tuple: (test_loss, metrics, predictions, ground_truth, inference_time)
    """
    
    model.eval()
    
    inference_start = time.time()

    test_loss = 0
    all_true = []
    all_pred = []
    all_prob = []

    with torch.no_grad():

        for images, labels in tqdm(test_loader, desc="Evaluating"):

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_true.extend(labels.cpu().numpy())
            all_pred.extend(preds.cpu().numpy())
            all_prob.extend(probs.cpu().numpy())

    test_loss /= len(test_loader)
    metrics = calculate_metrics(all_true, all_pred, all_prob)
    
    inference_time = time.time() - inference_start

    return (
    test_loss,
    metrics,
    all_pred,
    all_true,
    all_prob,
    inference_time
)


def main():
    """Main testing function."""
    
    # =====================================
    # SETUP
    # =====================================
    
    set_seed(config.SEED)
    create_result_directories()

    print(f"Device: {config.DEVICE}")

    # =====================================
    # LOAD DATA
    # =====================================

    print("\nLoading test dataset...")
    train_loader, val_loader, test_loader, class_names = get_dataloaders()

    print(f"Classes: {class_names}")

    # =====================================
    # LOAD MODEL
    # =====================================

    print("\nLoading model...")
    model = CustomModel(num_classes=config.NUM_CLASSES)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, weights_only=True))
    model = model.to(config.DEVICE)
    model.eval()

    target_layer_gradcam = model.conv_block[-2]
    target_layer_feature = model.conv_block
    target_layer_attention = model.amsfe_stage2

    print(f"Model loaded from: {config.MODEL_SAVE_PATH}")


    import os

    os.makedirs(
        os.path.join(
            config.TEST_RESULT_DIR,
            "gradcam"
        ),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(
            config.TEST_RESULT_DIR,
            "feature_maps"
        ),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(
            config.TEST_RESULT_DIR,
            "attention_maps"
        ),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(
            config.TEST_RESULT_DIR,
            "tsne"
        ),
        exist_ok=True
    )
    
    # Display model information
    print_model_info(model)

    # =====================================
    # LOSS FUNCTION
    # =====================================

    criterion = nn.CrossEntropyLoss()

    # =====================================
    # EVALUATE
    # =====================================

    print("\nEvaluating on test set...")
    test_loss, metrics, all_pred, all_true, all_prob, inference_time = evaluate(
        model,
        test_loader,
        criterion,
        config.DEVICE,
        class_names
    )

    # =====================================
    # SAVE RESULTS
    # =====================================

    print("\nSaving results...")
    save_test_metrics(test_loss, metrics)
    print_test_results(test_loss, metrics)

    save_complexity_report(
    model=model,
    accuracy=metrics["accuracy"],
    device=config.DEVICE
    )
    
    print(f"\nInference Time: {format_time(inference_time)}")
    print(f"Avg Time/Sample: {(inference_time / len(test_loader.dataset)) * 1000:.2f} ms")

    print("\nGenerating visualizations...")
    plot_confusion_matrix(all_true, all_pred, class_names)
    save_classification_report(all_true, all_pred, class_names)
    generate_tsne(
    model=model,
    dataloader=test_loader,
    device=config.DEVICE,
    save_path=os.path.join(
        config.TEST_RESULT_DIR,
        "tsne",
        "tsne.png"
    )
 )
    sample_image = None

    for images, labels in test_loader:

        sample_image = images[0].unsqueeze(0).to(
            config.DEVICE
        )

        break

    image_rgb = (
    sample_image[0]
    .permute(1, 2, 0)
    .cpu()
    .numpy()
    )

    image_rgb = (
        image_rgb - image_rgb.min()
    )

    image_rgb = (
        image_rgb / image_rgb.max()
    )

    generate_gradcam(
    model=model,
    image_tensor=sample_image,
    image_rgb=image_rgb,
    target_layer=target_layer_gradcam,
    save_path=os.path.join(
        config.TEST_RESULT_DIR,
        "gradcam",
        "gradcam.png"
    )
)
    
    save_feature_maps(
    model=model,
    image_tensor=sample_image,
    target_layer=target_layer_feature,
    save_dir=os.path.join(
        config.TEST_RESULT_DIR,
        "feature_maps"
    )
)
    
    save_attention_maps(
    model=model,
    image_tensor=sample_image,
    target_layer=target_layer_attention,
    save_dir=os.path.join(
        config.TEST_RESULT_DIR,
        "attention_maps"
    )
)
    
    plot_multiclass_roc(all_true, all_prob, class_names)
    plot_multiclass_pr(all_true, all_prob, class_names)
    save_error_analysis(all_true, all_pred, class_names)

    save_classwise_metrics(
        all_true,
        all_pred,
        class_names
    )

    plot_multiclass_roc(
    all_true,
    all_prob,
    class_names
    )

    plot_multiclass_pr(
        all_true,
        all_prob,
        class_names
    )

    save_error_analysis(
        all_true,
        all_pred,
        class_names
    )

    print("\nTesting complete!")
    print(f"Results saved to: {config.TEST_RESULT_DIR}/")


if __name__ == "__main__":
    main()