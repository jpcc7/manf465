from ultralytics import YOLO
import os
import shutil

# Checks if all images have a corresponding label
def check_sync(image_dir, label_dir):
    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        print("Error: Directory paths do not exist.")
        return False
        
    images = {os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png'))}
    labels = {os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.lower().endswith('.txt')}
    
    missing_labels = images - labels
    if missing_labels:
        print(f"Warning: {len(missing_labels)} images are missing labels!")
        return False
    
    print("All images are correctly associated with a label file.")
    return True

def start_training(clean=False):
    project_path = "models/fuse_counting_v1"
    trained_dir = "models/trained"
    
    # 1. Cleaning logic
    if clean and os.path.exists(project_path):
        print(f"Cleaning out old session at {project_path}...")
        shutil.rmtree(project_path)
        
        for root, dirs, files in os.walk("data"):
            for file in files:
                if file.endswith(".cache"):
                    os.remove(os.path.join(root, file))

    # 2. Training logic
    model = YOLO("yolo11n.pt")
    results = model.train(
        data="data/dataset.yaml",
        epochs=5,
        imgsz=640,
        device="cpu", # Changed to CPU as per your snippet
        project="models",
        name="fuse_counting_v1",
        exist_ok=True 
    )

    # 3. Post-training: Export weights to models/trained
    os.makedirs(trained_dir, exist_ok=True)
    source_weights = os.path.join(project_path, "weights", "best.pt")
    target_weights = os.path.join(trained_dir, "fuse_v1.pt")

    if os.path.exists(source_weights):
        shutil.copy(source_weights, target_weights)
        print(f"--- SUCCESS ---")
        print(f"Model saved to: {target_weights}")
    else:
        print("Error: Training completed but best.pt was not found.")

if __name__ == "__main__":
    if check_sync("data/train/images", "data/train/labels"):
        start_training(clean=True) # Set to True to ensure a fresh weights copy
    else:
        print("Training aborted due to missing labels.")