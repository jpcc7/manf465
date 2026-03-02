from ultralytics import YOLO
import os

# Checks if all images have a corresponding label
def check_sync(image_dir, label_dir):
    images = {os.path.splitext(f)[0] for f in os.listdir(image_dir)}
    labels = {os.path.splitext(f)[0] for f in os.listdir(label_dir)}
    
    missing_labels = images - labels
    if missing_labels:
        print(f"Warning: {len(missing_labels)} images are missing labels!")
        return 0
    else:
        print("All images are correctly associated with a label file.")
        return 1

# Loads data into YOLO model for training
def train_model():
    # Load pretrained YOLO11 Nano model
    model = YOLO("yolo11n.pt") 

    # Fine-tune the model
    # We use a small number of epochs because your environment is static
    results = model.train(
        data="data/dataset.yaml",
        epochs=50,           
        imgsz=640,           # Standard resolution
        device="cpu",        # Local device for training
        project="models",    # Where to save the output
        name="conveyor_v1",
        plots=True           # Generates accuracy/loss charts automatically
    )
    
    print("Training complete! Model saved in models/conveyor_v1/weights/best.pt")

# Usage
if __name__ == "__main__":
    check_sync("data/train/images", "data/train/labels")
    train_model()