import cv2
import numpy as np
from picamera2 import Picamera2
from picamera2.previews import NullPreview
from ultralytics import YOLO

# 1. Load your YOLO model
model = YOLO("fuse_v1.pt")

# 2. Initialize Picamera2 - The official 2026 Pi Camera API
picam2 = Picamera2()
# Configure for a stable 640x480 stream
config = picam2.create_video_configuration(main={"format": "RGB888", "size": (640, 480)})
picam2.configure(config)
picam2.start_preview(Picamera2.NullPreview())
picam2.start()

print("Starting Real-Time Inference (Picamera2)... Press 'q' to quit.")

try:
    while True:
        # 3. Capture a frame directly as a numpy array
        # This bypasses the V4L2 'reshape' bug entirely
        frame = picam2.capture_array()

        # 4. Convert RGB (Picamera2 default) to BGR (OpenCV default)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 5. Run YOLO Inference
        # imgsz=320 keeps the FPS high on your Pi 4
        results = model(frame_bgr, imgsz=320, conf=0.5, verbose=False)

        # 6. Visualize
        annotated_frame = results[0].plot()
        
        # 7. Show in VNC window
        cv2.imshow("YOLO11 Fuse Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    # Proper hardware cleanup
    picam2.stop()
    cv2.destroyAllWindows()