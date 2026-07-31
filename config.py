import os
import torch

# ============================================
# DATASET PATHS
# ============================================

DATASET_PATH = "Gallblader"
TRAIN_DIR = os.path.join(DATASET_PATH, "train")
VAL_DIR = os.path.join(DATASET_PATH, "val")
TEST_DIR = os.path.join(DATASET_PATH, "test")

# ============================================
# TRAINING HYPERPARAMETERS
# ============================================

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 9

NUM_WORKERS = 4

# ============================================
# DEVICE
# ============================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================
# RESULTS DIRECTORIES
# ============================================

TRAIN_RESULT_DIR = "train_results"
TEST_RESULT_DIR = "test_results"

MODEL_SAVE_PATH = os.path.join(TRAIN_RESULT_DIR, "best_model.pth")

# ============================================
# MODEL VARIANT SELECTION (for ablation study)
# ============================================
# Change ONLY this one line to switch between the full model and any
# ablation variant. train.py, test.py, complexity.py and allresult.py all
# read from this automatically -- no other file needs editing, and no
# --network-file / --checkpoint flags are needed on the command line.
#
#   "full"      -> network.py      (full ACMS-Net)
#   "ablation1" -> ablation1.py   (AMSFE only)
MODEL_VARIANT = "ablation1"

VARIANT_REGISTRY = {
    "full": {
        "network_file": "network.py",
        "checkpoint": os.path.join(TRAIN_RESULT_DIR, "best_model_full.pth"),
    },
    "ablation1": {
        "network_file": "ablation1.py",
        "checkpoint": os.path.join(TRAIN_RESULT_DIR, "best_model_ablation1.pth"),
    },
}

if MODEL_VARIANT not in VARIANT_REGISTRY:
    raise ValueError(
        f"config.MODEL_VARIANT={MODEL_VARIANT!r} is not in VARIANT_REGISTRY. "
        f"Valid options: {list(VARIANT_REGISTRY)}"
    )

NETWORK_FILE = VARIANT_REGISTRY[MODEL_VARIANT]["network_file"]

# MODEL_SAVE_PATH now automatically points to a variant-specific checkpoint,
# so training the full model and an ablation variant never overwrite each
# other's weights.
MODEL_SAVE_PATH = VARIANT_REGISTRY[MODEL_VARIANT]["checkpoint"]

# ============================================
# SEEDS
# ============================================

SEED = 42

# ============================================
# SCHEDULER PARAMETERS
# ============================================

SCHEDULER_MODE = 'min'
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 0.5