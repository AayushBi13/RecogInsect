import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort.deep_sort import DeepSort

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Initialize DeepSORT tracker
deepsort = DeepSort(model_path="deep_sort/deep/checkpoint/ckpt.t7", max_age=30, n_init=3, nms_max_overlap=1.0)

# Open webcam or video file
cap = cv2.VideoCapture(0)  # Change to 'video.mp4' for video file

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform YOLO detection
    results = model(frame)
    
    detections = []
    confidences = []
    
    for r in results:
        for box in r.boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()
            if conf > 0.5:  # Confidence threshold
                detections.append([x1, y1, x2 - x1, y2 - y1])  # Convert to [x, y, w, h] format
                confidences.append(conf)

    # Convert detections to numpy array
    if len(detections) > 0:
        detections = np.array(detections)
        confidences = np.array(confidences)
    else:
        detections = np.empty((0, 4))
        confidences = np.empty((0,))

    # Update DeepSORT tracker
    track_ids = deepsort.update(detections, confidences, frame)

    # Draw tracking boxes
    for track in track_ids:
        x, y, w, h, track_id = track.astype(int)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display frame
    cv2.imshow('YOLO + DeepSORT Tracking', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
