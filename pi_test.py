import cv2
import time
from ultralytics import YOLO

# 1. Load the model
model = YOLO("fuse_v1.pt")

# 2. Simplified Camera Init
# Removing the manual FOURCC and resolution fixes the 'reshape' error
cap = cv2.VideoCapture(0) 

print("Starting Real-Time Inference... Press 'q' to quit.")
prev_time = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret or frame is None:
        # If we get a single bad frame, just skip it
        continue

    # 3. Run Inference (imgsz=320 is still key for Pi 4 speed)
    results = model(frame, imgsz=320, stream=True, conf=0.5)

    # 4. Process Results
    for r in results:
        annotated_frame = r.plot()

    # 5. FPS Logic
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 6. Show the Window
    cv2.imshow("YOLO11 Fuse Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()