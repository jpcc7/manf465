import os
from pathlib import Path

# --- BASE DIRECTORY ---
# This points to the manf465/ root folder
BASE_DIR = Path(__file__).resolve().parent.parent

# --- DATA PATHS ---
DATA_DIR = BASE_DIR / "data"
DATA_YAML = DATA_DIR / "dataset.yaml"

TRAIN_IMG_DIR = DATA_DIR / "train" / "images"
TRAIN_LBL_DIR = DATA_DIR / "train" / "labels"
VAL_IMG_DIR = DATA_DIR / "val" / "images"
VAL_LBL_DIR = DATA_DIR / "val" / "labels"

# --- MODEL PATHS ---
MODELS_DIR = BASE_DIR / "models"
TRAINING_PROJECT_DIR = MODELS_DIR / "training_runs" # Keeps logs organized
TRAINED_EXPORT_DIR = MODELS_DIR / "trained"
FINAL_MODEL_PATH = TRAINED_EXPORT_DIR / "fuse_v1.pt"

# --- TRAINING SETTINGS ---
EPOCHS = 30
IMGSZ = 640
DEVICE = "cpu" # Switch to "cuda" when you're back on GPU