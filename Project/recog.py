from ultralytics import YOLO
import cv2
model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        break
    results = model(frame)
    for r in results:
        frame = r.plot()
    cv2.imshow('YOLO Insect Detection', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
     