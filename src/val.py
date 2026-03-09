from ultralytics import YOLO
import os

def run_validation():
    # Path to your newly moved weights
    model_path = os.path.join("models", "trained", "fuse_v1.pt")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    model = YOLO(model_path)
    
    # Run validation using the config in your data folder
    metrics = model.val(data="data/dataset.yaml")
    
    print(f"Validation Complete.")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")

if __name__ == "__main__":
    run_validation()