from ultralytics import YOLO
import cv2
import numpy as np
from sort import Sort 


model = YOLO('yolov8n.pt')


tracker = Sort()


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

   
    results = model(frame)

    detections = []
    for r in results:
        for box in r.boxes: 
            x1, y1, x2, y2 = box.xyxy[0].tolist() 
            score = box.conf[0].item()  
            cls = box.cls[0].item()  

            if score > 0.5:
                detections.append([x1, y1, x2, y2, score])


    if len(detections) > 0:
        detections = np.array(detections)
    else:
        detections = np.empty((0, 5))  


    track_ids = tracker.update(detections)


    for track in track_ids:
        x1, y1, x2, y2, track_id = track.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    cv2.imshow('YOLO Detection', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
