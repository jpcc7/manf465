import cv2
from ultralytics import YOLO

# 1. Load your model
model = YOLO("fuse_v1.pt")

# 2. Simple Camera Init
# We rely on the v4l2-ctl command we ran earlier to set the resolution
cap = cv2.VideoCapture(0)

print("Starting inference... Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()

    # 3. Robust Frame Check
    # This prevents the 'reshape' error by ensuring the frame isn't empty
    if not ret or frame is None or frame.size == 0:
        continue

    try:
        # 4. Run YOLO inference
        # imgsz=320 keeps it fast on the Pi 4
        results = model(frame, imgsz=320, conf=0.5, verbose=False)

        # 5. Visualize and Show
        # results[0].plot() creates the image with bounding boxes
        annotated_frame = results[0].plot()
        
        cv2.imshow("Fuse Detection Test", annotated_frame)

    except Exception as e:
        print(f"Inference error: {e}")
        continue

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()