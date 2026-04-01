import cv2
import numpy as np
from picamera2 import Picamera2
# Correct the import for the preview module
from picamera2.previews import NullPreview
from ultralytics import YOLO

# 1. Load the model
model = YOLO("fuse_v1.pt")

# 2. Initialize Picamera2
picam2 = Picamera2()

# 3. Configure the stream
config = picam2.create_video_configuration(main={"format": "RGB888", "size": (640, 480)})
picam2.configure(config)

# 4. FIX: Use NullPreview to bypass the 'pykms' dependency error
picam2.start_preview(NullPreview())
picam2.start()

print("Starting Inference... Look at your VNC window for the output.")

try:
    while True:
        # Capture frame as array
        frame = picam2.capture_array()

        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Run YOLO Inference (imgsz=320 for speed)
        results = model(frame_bgr, imgsz=320, conf=0.5, verbose=False)

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                conf_score = float(box.conf[0])
                name = model.names[class_id]
                print(f"Detected: {name} | Confidence: {conf_score:.2f}")

        # Annotate and show
        annotated_frame = results[0].plot()
        cv2.imshow("Fuse Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    picam2.stop()
    cv2.destroyAllWindows()