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
            bbox = box.xyxy[0].tolist()
            if len(bbox) < 4:
                continue  

            x1, y1, x2, y2 = map(int, bbox)
            score = float(box.conf[0])
            cls = int(box.cls[0])  

            if score > 0.5:
                detections.append([x1, y1, x2, y2, score, cls])  


    detections = np.array(detections) if detections else np.empty((0, 6))
    track_ids = tracker.update(detections[:, :5])  

    for i, track in enumerate(track_ids):
        x1, y1, x2, y2, tid = track.astype(int)
        
        obj_class = int(detections[i, 5]) 

        
        
        if obj_class == 0:  
            cx,cy = (x1+x2)//2, (y1+y2)//2
            
            color = (0,255,0) 
        
            cv2.line(frame,(cx-20,cy),(cx+20,cy),color,2)
            cv2.line(frame,(cx,cy-20),(cx,cy+20),color,2)

        cv2.putText(frame, f'ID {tid}', (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('Object Detection', frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break  

cap.release()
cv2.destroyAllWindows()
