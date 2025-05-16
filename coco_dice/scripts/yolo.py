import cv2
import numpy as np
from ultralytics import YOLO

def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def is_above(point1, point2, threshold=30):
    return point2[1] - point1[1] > threshold

def is_below(point1, point2, threshold=15):
    return 0 < abs(point1[1] - point2[1]) <= threshold

def is_at_same_height(point1, point2, threshold=30):
    return 0 < abs(point1[1] - point2[1]) < threshold

def is_right_of(point1, point2, threshold=30):
    return 0 < abs(point1[0] - point2[0]) < threshold

def is_left_of(point1, point2, threshold=30):
    return point2[0] - point1[0] > threshold

def is_near(point1, point2, threshold=50):
    return calculate_distance(point1, point2) < threshold

NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12

def detect_poses(keypoints):
    poses = []
    
    if (is_above(keypoints[RIGHT_WRIST], keypoints[RIGHT_SHOULDER]) and 
        is_above(keypoints[RIGHT_ELBOW], keypoints[RIGHT_SHOULDER]) and
        keypoints[RIGHT_WRIST][1] != 0 and keypoints[RIGHT_ELBOW][1] != 0):
        poses.append("Brazo derecho arriba")
    
    if (is_above(keypoints[LEFT_WRIST], keypoints[LEFT_SHOULDER]) and 
        is_above(keypoints[LEFT_ELBOW], keypoints[LEFT_SHOULDER]) and
        keypoints[LEFT_WRIST][1] != 0 and keypoints[LEFT_ELBOW][1] != 0):
        poses.append("Brazo izquierdo arriba")
    
    if "Brazo derecho arriba" in poses and "Brazo izquierdo arriba" in poses:
        poses.append("Ambos brazos arriba")
    
    if (is_below(keypoints[RIGHT_WRIST], keypoints[RIGHT_HIP]) and 
        is_below(keypoints[LEFT_WRIST], keypoints[LEFT_HIP])):
        poses.append("Ambos brazos abajo")
    
    if (is_at_same_height(keypoints[RIGHT_WRIST], keypoints[RIGHT_SHOULDER], 100) and 
        is_right_of(keypoints[RIGHT_WRIST], keypoints[RIGHT_SHOULDER], 100)):
        poses.append("Brazo derecho hacia delante")

    if (is_at_same_height(keypoints[LEFT_WRIST], keypoints[LEFT_SHOULDER], 100) and 
        is_right_of(keypoints[LEFT_WRIST], keypoints[LEFT_SHOULDER], 100)):
        poses.append("Brazo izquierdo hacia delante")

    if "Brazo derecho hacia delante" in poses and "Brazo izquierdo hacia delante" in poses:
        poses.append("Ambos brazos hacia delante")
    
    # if ((is_above(keypoints[RIGHT_WRIST], keypoints[RIGHT_SHOULDER]) and 
    #      is_above(keypoints[LEFT_WRIST], keypoints[LEFT_SHOULDER])) and
    #     (is_left_of(keypoints[RIGHT_WRIST], keypoints[RIGHT_SHOULDER]) and 
    #      is_right_of(keypoints[LEFT_WRIST], keypoints[LEFT_SHOULDER]))):
    #     poses.append("Símbolo X con los brazos")
    
    # if is_near(keypoints[RIGHT_WRIST], keypoints[NOSE]):
    #     poses.append("Tocando nariz con mano derecha")
    
    # if is_near(keypoints[LEFT_WRIST], keypoints[NOSE]):
    #     poses.append("Tocando nariz con mano izquierda")
    
    # if "Tocando nariz con mano derecha" in poses and "Tocando nariz con mano izquierda" in poses:
    #     poses.append("Tocando nariz con ambas manos")
    
    # if is_near(keypoints[LEFT_WRIST], keypoints[LEFT_EYE]):
    #     poses.append("Tocando ojo izquierdo con mano izquierda")

    # if is_near(keypoints[RIGHT_WRIST], keypoints[LEFT_EYE]):
    #     poses.append("Tocando ojo izquierdo con mano derecha")
    
    # if ("Tocando ojo izquierdo con mano izquierda" in poses and 
    #     "Tocando ojo izquierdo con mano derecha" in poses):
    #     poses.append("Tocando ojo izquierdo con ambas manos")
    
    # if is_near(keypoints[RIGHT_WRIST], keypoints[RIGHT_EYE]):
    #     poses.append("Tocando ojo derecho con mano derecha")

    # if is_near(keypoints[LEFT_WRIST], keypoints[RIGHT_EYE]):
    #     poses.append("Tocando ojo derecho con mano izquierda")

    # if ("Tocando ojo derecho con mano derecha" in poses and 
    #     "Tocando ojo derecho con mano izquierda" in poses):
    #     poses.append("Tocando ojo derecho con ambas manos")
    
    # if is_near(keypoints[RIGHT_WRIST], keypoints[RIGHT_EAR]):
    #     poses.append("Tocando oreja derecha con mano derecha")

    # if is_near(keypoints[LEFT_WRIST], keypoints[RIGHT_EAR]):
    #     poses.append("Tocando oreja derecha con mano izquierda")

    # if ("Tocando oreja derecha con mano derecha" in poses and 
    #     "Tocando oreja derecha con mano izquierda" in poses):
    #     poses.append("Tocando oreja derecha con ambas manos")
    
    # if is_near(keypoints[LEFT_WRIST], keypoints[LEFT_EAR]):
    #     poses.append("Tocando oreja izquierda con mano izquierda")

    # if is_near(keypoints[RIGHT_WRIST], keypoints[LEFT_EAR]):
    #     poses.append("Tocando oreja izquierda con mano derecha")

    # if ("Tocando oreja izquierda con mano izquierda" in poses and 
    #     "Tocando oreja izquierda con mano derecha" in poses):
    #     poses.append("Tocando oreja izquierda con ambas manos")
    
    # if is_near(keypoints[LEFT_WRIST], keypoints[LEFT_SHOULDER]):
    #     poses.append("Tocando hombro izquierdo con mano izquierda")
    
    # if is_near(keypoints[RIGHT_WRIST], keypoints[LEFT_SHOULDER]):
    #     poses.append("Tocando hombro izquierdo con mano derecha")
    
    # if is_near(keypoints[RIGHT_WRIST], keypoints[RIGHT_SHOULDER]):
    #     poses.append("Tocando hombro derecho con mano derecha")
    
    # if is_near(keypoints[LEFT_WRIST], keypoints[RIGHT_SHOULDER]):
    #     poses.append("Tocando hombro derecho con mano izquierda")
    
    # if is_near(keypoints[RIGHT_WRIST], keypoints[LEFT_ELBOW]):
    #     poses.append("Tocando codo izquierdo con mano derecha")
    
    # if is_near(keypoints[LEFT_WRIST], keypoints[RIGHT_ELBOW]):
    #     poses.append("Tocando codo derecho con mano izquierda")
    
    # if (is_above(keypoints[RIGHT_WRIST], keypoints[RIGHT_SHOULDER]) and 
    #     is_above(keypoints[RIGHT_WRIST], keypoints[RIGHT_ELBOW]) and
    #     is_near(keypoints[RIGHT_WRIST], keypoints[RIGHT_EAR], 100)):
    #     poses.append("Saludo con mano derecha")
    
    # if (is_above(keypoints[LEFT_WRIST], keypoints[LEFT_SHOULDER]) and 
    #     is_above(keypoints[LEFT_WRIST], keypoints[LEFT_ELBOW]) and
    #     is_near(keypoints[LEFT_WRIST], keypoints[LEFT_EAR], 100)):
    #     poses.append("Saludo con mano izquierda")
    
    return poses

model = YOLO("../models/yolov8m-pose.pt")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

cv2.namedWindow("Detector de Poses", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    annotated_frame = results[0].plot()

    keypoints = results[0].keypoints
    if keypoints is not None and len(keypoints.xy) > 0:
        person_keypoints = []
        for j, (x, y) in enumerate(keypoints.xy[0].cpu().numpy()):
            person_keypoints.append((float(x), float(y)))
        
        poses = detect_poses(person_keypoints)
        
        y_pos = 30
        for pose in poses:
            cv2.putText(annotated_frame, pose, (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_pos += 30
        
    cv2.imshow("Detector de Poses", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
