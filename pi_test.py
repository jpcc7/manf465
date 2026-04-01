import cv2
import time
from ultralytics import YOLO

# 1. Load your trained model
# Ensure 'fuse_v1.pt' is in the same folder as this script
model = YOLO("fuse_v1.pt")

# 2. Initialize Camera
# 0 is usually the ribbon cable camera, 1 is often a USB webcam
cap = cv2.VideoCapture(0)

# OPTIMIZATION: Set a lower resolution to reduce the workload on the Pi CPU
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Starting Real-Time Inference... Press 'q' to quit.")

# Variables for FPS calculation
prev_time = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Error: Could not read from camera.")
        break

    # 3. Run Inference
    # imgsz=320: Downscaling the image for the AI makes it 2-3x faster
    # stream=True: Uses a generator to save RAM
    # conf=0.5: Only show detections with >50% certainty
    results = model(frame, imgsz=320, stream=True, conf=0.5)

    # 4. Process Results
    for r in results:
        # Draw bounding boxes and labels onto the frame
        annotated_frame = r.plot()

        # LOGIC: Count how many fuses are currently in the frame
        fuse_count = len(r.boxes)
        
        # Display the count on the terminal for debugging
        if fuse_count > 0:
            print(f"Fuses detected: {fuse_count}")

    # 5. Calculate and Display FPS (Frames Per Second)
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 6. Show the Window (Must be in VNC to see this)
    cv2.imshow("MANF 465 - Real-Time Fuse Detection", annotated_frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()