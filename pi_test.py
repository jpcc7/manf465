import cv2
import time
from ultralytics import YOLO

# 1. Load the model
# Make sure 'fuse_v1.pt' is in the same folder as this script
model = YOLO("fuse_v1.pt")

# 2. Initialize the camera using the V4L2 backend
# Index 0 is the standard for the primary CSI camera
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# 3. Hardware Optimization
# Setting the format to MJPG prevents 'NoneType' frames on newer Pi kernels
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Starting Real-Time Inference... Press 'q' to quit.")

prev_time = 0

while cap.isOpened():
    # Capture the frame
    ret, frame = cap.read()

    # --- THE FIX ---
    # If the camera handshake drops for a millisecond, 'ret' will be False.
    # This check prevents trying to process 'None', which causes the 'reshape' error.
    if not ret or frame is None:
        continue

    # 4. Run Inference
    # imgsz=320 is essential for speed on Pi 4
    # stream=True optimizes memory usage
    results = model(frame, imgsz=320, stream=True, conf=0.5)

    # 5. Process and Visualize Results
    for r in results:
        annotated_frame = r.plot()
        
        # Terminal log for debugging
        detect_count = len(r.boxes)
        if detect_count > 0:
            print(f"Fuses detected: {detect_count}")

    # 6. Calculate and Display FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 7. Show the Window (Must be inside VNC)
    cv2.imshow("YOLO11 Fuse Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()