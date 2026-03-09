from ultralytics import YOLO

# Load fine-tuned model
model = YOLO("models/conveyor_v1/weights/best.pt")

def process_frame(frame):
    # Run inference
    results = model(frame)
    
    # The 'results' object contains a list of detections
    # Each detection is an instance of a 'fuse'
    fuse_count = len(results[0].boxes)
    
    if fuse_count == 0:
        print("Status: 0 Fuses (Empty/Reject)")
    elif fuse_count == 1:
        print("Status: 1 Fuse (Incomplete)")
    elif fuse_count == 2:
        print("Status: 2 Fuses (Complete)")
    else:
        print(f"Status: Alert! Found {fuse_count} fuses.")

    return results[0].plot() # Returns image with boxes drawn


# Run inference on a specific test image
results = model.predict(source="data/label_export/images/0dc031ac-one_fuse_013.jpg", save=True)

# Print how many fuses it found
for result in results:
    print(f"Detected {len(result.boxes)} fuses in this image.")