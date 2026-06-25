import cv2
import numpy as np
from ultralytics import YOLO
from sort.sort import Sort

# Load Yolo Model
model = YOLO("yolov8s.pt")
class_names = model.names

# Tracker

tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.3)

# Video input

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    # YOLO detection
    results = model(frame, verbose=False)

    detections = []
    detection_labels = {}

    idx = 0

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if conf > 0.5:
                label = class_names[cls]

                detections.append([x1, y1, x2, y2, conf])
                detection_labels[idx] = label
                idx += 1

    detections = np.array(detections)

    if len(detections) == 0:
        detections = np.empty((0, 5))

    # SORT tracking
    tracks = tracker.update(detections)

    # Draw results
    for i, track in enumerate(tracks):
        x1, y1, x2, y2, track_id = track.astype(int)

        label = detection_labels.get(i, "object")

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame,
                    f"{label} ID:{track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2)

    cv2.imshow("Smart Object Detection + Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()