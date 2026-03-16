from ultralytics import YOLO
import os
import shutil
from config import * # Imports BASE_DIR, DATA_DIR, FINAL_MODEL_PATH, etc.

# Checks if all images have a corresponding label
def check_sync(image_dir, label_dir):
    # Pathlib objects (from config) support .exists() natively
    if not image_dir.exists() or not label_dir.exists():
        print(f"Error: Directory paths do not exist: {image_dir} or {label_dir}")
        return False
        
    # Using Pathlib's glob or iterdir is cleaner than os.listdir
    images = {f.stem for f in image_dir.iterdir() if f.suffix.lower() in ('.jpg', '.png')}
    labels = {f.stem for f in label_dir.glob("*.txt")}
    
    missing_labels = images - labels
    if missing_labels:
        print(f"Warning: {len(missing_labels)} images in {image_dir.name} are missing labels!")
        return False
    
    print(f"Sync Check Passed for {image_dir.parent.name}/{image_dir.name}.")
    return True

def start_training(clean=True):
    # 1. Cleaning logic using paths from config.py
    if clean:
    # Pre-Training Cleanup
        print("Performing pre-training cleanup...")
        if MODELS_DIR.exists():
            for item in MODELS_DIR.iterdir():
                # Delete everything EXCEPT the 'trained' folder
                if item.is_dir() and item.name != "trained":
                    shutil.rmtree(item)
                elif item.is_file():
                    os.remove(item)
    
    # Also wipe YOLO caches
        for cache_file in DATA_DIR.rglob("*.cache"):
            os.remove(cache_file)

    # 2. Training logic
    model = YOLO("yolo11n.pt")

    model.train(
        data=str(DATA_YAML),      # YOLO expects strings, so we cast Pathlib objects
        epochs=EPOCHS,            # Using value from config.py
        imgsz=IMGSZ,              # Using value from config.py
        device=DEVICE,            # Using value from config.py
        project=str(MODELS_DIR),
        name=TRAINING_PROJECT_DIR.name,
        exist_ok=True 
    )

    # 3. Post-training: Export weights to models/trained/fuse_v1.pt
    TRAINED_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    # YOLO saves weights inside the project/name/weights/ folder
    source_best = TRAINING_PROJECT_DIR / "weights" / "best.pt"
    source_last = TRAINING_PROJECT_DIR / "weights" / "last.pt"

    if source_best.exists():
        shutil.copy(source_best, FINAL_MODEL_PATH)
        print(f"--- SUCCESS ---")
        print(f"Best model saved to: {FINAL_MODEL_PATH}")
    elif source_last.exists():
        shutil.copy(source_last, FINAL_MODEL_PATH)
        print(f"--- SUCCESS (Using Last) ---")
        print(f"Last model saved to: {FINAL_MODEL_PATH}")
    else:
        print(f"Error: No weights found in {TRAINING_PROJECT_DIR}/weights/")

if __name__ == "__main__":
    # Check both Train and Val sets using the clean paths from config.py
    if check_sync(TRAIN_IMG_DIR, TRAIN_LBL_DIR) and check_sync(VAL_IMG_DIR, VAL_LBL_DIR):
        start_training(clean=True)
    else:
        print("Training aborted due to synchronization issues.")