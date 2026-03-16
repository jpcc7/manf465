from ultralytics import YOLO
import cv2
import os
from config import FINAL_MODEL_PATH, DATA_DIR

# 1. Load fine-tuned model using Path object from config
if not FINAL_MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {FINAL_MODEL_PATH}. Run main.py first!")

# YOLO handles Pathlib objects natively, but casting to str is safest for all versions
model = YOLO(str(FINAL_MODEL_PATH))

def process_frame(frame, conf=0.5):
    """
    Processes a single frame (numpy array) and returns the count + annotated image.
    Used for both static test images and live camera streams.
    """
    # Run inference
    results = model(frame, conf=conf)
    result = results[0]
    fuse_count = len(result.boxes)
    
    # Logic for MANF 465 Conveyor System
    if fuse_count == 0:
        status = "0 Fuses (Empty/Reject)"
    elif fuse_count == 1:
        status = "1 Fuse (Incomplete)"
    elif fuse_count == 2:
        status = "2 Fuses (Complete)"
    else:
        status = f"Alert! Found {fuse_count} fuses."

    print(f"[{status}] - Detections: {fuse_count}")
    
    # Returns the image with bounding boxes drawn
    return result.plot(), fuse_count

# --- TEST EXECUTION ---
if __name__ == "__main__":
    # Point to an image in your label_export directory using the config's DATA_DIR
    test_image_path = DATA_DIR / "raw" / "two_fuse" / "two_fuse_000.jpg"
    
    if test_image_path.exists():
        # Load image using OpenCV
        img = cv2.imread(str(test_image_path))
        
        # Process frame
        annotated_img, count = process_frame(img, conf=0.6)
        
        # Save output to the project root for verification
        output_path = test_image_path.parent.parent.parent.parent / "test_result.jpg"
        cv2.imwrite(str(output_path), annotated_img)
        
        print(f"--- SUCCESS ---")
        print(f"Processed: {test_image_path.name}")
        print(f"Visual result saved to: {output_path}")
    else:
        print(f"Error: Test image not found at {test_image_path}")