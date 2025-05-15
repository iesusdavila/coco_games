import cv2
from ultralytics import YOLO

cv2.startWindowThread() 

model = YOLO("../models/yolov8m-pose.pt")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

cv2.namedWindow("Cámara YOLO", cv2.WINDOW_NORMAL) 
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    keypoints = results[0].keypoints
    if keypoints is not None:
        person = keypoints.xy[0]
        for j, (x, y) in enumerate(person):
            print(f"    Punto {j}: x={x:.1f}, y={y:.1f}")

    annotated_frame = cv2.flip(annotated_frame, 1)  
    cv2.imshow("Cámara YOLO", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()





