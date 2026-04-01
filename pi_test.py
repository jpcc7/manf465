import cv2
from ultralytics import YOLO

# 1. Load the model
model = YOLO("fuse_v1.pt")

# 2. Open Camera 0
# We don't set ANY properties here. Let the OS handle the format.
cap = cv2.VideoCapture(0)

print("Starting inference... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()

    # 3. Validation: Stop the 'reshape' error before it starts
    if not ret or frame is None:
        continue

    # 4. YOLO Inference
    # verbose=False cleans up your terminal output
    results = model(frame, imgsz=320, conf=0.5, verbose=False)

    # 5. Display
    annotated_frame = results[0].plot()
    cv2.imshow("MANF 465 Fuse Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()