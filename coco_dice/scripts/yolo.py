import cv2
import numpy as np
import math
from ultralytics import YOLO

# Función para calcular la distancia euclidiana entre dos puntos
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# Función para determinar si un punto está por encima de otro
def is_above(point1, point2, threshold=30):
    return point2[1] - point1[1] > threshold

# Función para determinar si un punto está por debajo de otro
def is_below(point1, point2, threshold=30):
    return point1[1] - point2[1] > threshold

# Función para determinar si dos puntos están aproximadamente a la misma altura
def is_at_same_height(point1, point2, threshold=30):
    return abs(point1[1] - point2[1]) < threshold

# Función para determinar si un punto está a la derecha de otro
def is_right_of(point1, point2, threshold=30):
    return point1[0] - point2[0] > threshold

# Función para determinar si un punto está a la izquierda de otro
def is_left_of(point1, point2, threshold=30):
    return point2[0] - point1[0] > threshold

# Función para determinar si dos puntos están cerca entre sí
def is_near(point1, point2, threshold=50):
    return calculate_distance(point1, point2) < threshold

# Índices de los keypoints según el formato COCO usado por YOLO
# Nota: YOLO sigue el formato COCO para los keypoints
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
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

# Función para detectar las diferentes posiciones
def detect_poses(keypoints):
    poses = []
    
    # Verificar si tenemos todos los keypoints necesarios
    if keypoints is None or len(keypoints) < 17:
        return poses
    
    # Brazo derecho arriba
    if is_above(keypoints[RIGHT_WRIST], keypoints[RIGHT_SHOULDER]) and is_above(keypoints[RIGHT_ELBOW], keypoints[RIGHT_SHOULDER]):
        poses.append("Brazo derecho arriba")
    
    # Brazo izquierdo arriba
    if is_above(keypoints[LEFT_WRIST], keypoints[LEFT_SHOULDER]) and is_above(keypoints[LEFT_ELBOW], keypoints[LEFT_SHOULDER]):
        poses.append("Brazo izquierdo arriba")
    
    # Ambos brazos arriba
    if "Brazo derecho arriba" in poses and "Brazo izquierdo arriba" in poses:
        poses.append("Ambos brazos arriba")
    
    # Ambos brazos abajo
    if (is_below(keypoints[RIGHT_WRIST], keypoints[RIGHT_HIP]) and 
        is_below(keypoints[LEFT_WRIST], keypoints[LEFT_HIP])):
        poses.append("Ambos brazos abajo")
    
    # Ambos brazos hacia delante (asumiendo que hacia delante significa extendidos horizontalmente)
    if (is_at_same_height(keypoints[RIGHT_WRIST], keypoints[RIGHT_SHOULDER], 50) and 
        is_at_same_height(keypoints[LEFT_WRIST], keypoints[LEFT_SHOULDER], 50) and
        is_right_of(keypoints[RIGHT_WRIST], keypoints[RIGHT_SHOULDER]) and
        is_left_of(keypoints[LEFT_WRIST], keypoints[LEFT_SHOULDER])):
        poses.append("Ambos brazos hacia delante")
    
    # Símbolo X con los brazos
    if ((is_above(keypoints[RIGHT_WRIST], keypoints[RIGHT_SHOULDER]) and 
         is_above(keypoints[LEFT_WRIST], keypoints[LEFT_SHOULDER])) and
        (is_left_of(keypoints[RIGHT_WRIST], keypoints[RIGHT_SHOULDER]) and 
         is_right_of(keypoints[LEFT_WRIST], keypoints[LEFT_SHOULDER]))):
        poses.append("Símbolo X con los brazos")
    
    # Tocarse la nariz con mano derecha
    if is_near(keypoints[RIGHT_WRIST], keypoints[NOSE]):
        poses.append("Tocando nariz con mano derecha")
    
    # Tocarse la nariz con mano izquierda
    if is_near(keypoints[LEFT_WRIST], keypoints[NOSE]):
        poses.append("Tocando nariz con mano izquierda")
    
    # Tocarse la nariz con ambas manos
    if "Tocando nariz con mano derecha" in poses and "Tocando nariz con mano izquierda" in poses:
        poses.append("Tocando nariz con ambas manos")
    
    # Tocarse el ojo izquierdo
    if is_near(keypoints[LEFT_WRIST], keypoints[LEFT_EYE]):
        poses.append("Tocando ojo izquierdo con mano izquierda")
    if is_near(keypoints[RIGHT_WRIST], keypoints[LEFT_EYE]):
        poses.append("Tocando ojo izquierdo con mano derecha")
    if "Tocando ojo izquierdo con mano izquierda" in poses and "Tocando ojo izquierdo con mano derecha" in poses:
        poses.append("Tocando ojo izquierdo con ambas manos")
    
    # Tocarse el ojo derecho
    if is_near(keypoints[RIGHT_WRIST], keypoints[RIGHT_EYE]):
        poses.append("Tocando ojo derecho con mano derecha")
    if is_near(keypoints[LEFT_WRIST], keypoints[RIGHT_EYE]):
        poses.append("Tocando ojo derecho con mano izquierda")
    if "Tocando ojo derecho con mano derecha" in poses and "Tocando ojo derecho con mano izquierda" in poses:
        poses.append("Tocando ojo derecho con ambas manos")
    
    # Tocarse la oreja derecha
    if is_near(keypoints[RIGHT_WRIST], keypoints[RIGHT_EAR]):
        poses.append("Tocando oreja derecha con mano derecha")
    if is_near(keypoints[LEFT_WRIST], keypoints[RIGHT_EAR]):
        poses.append("Tocando oreja derecha con mano izquierda")
    if "Tocando oreja derecha con mano derecha" in poses and "Tocando oreja derecha con mano izquierda" in poses:
        poses.append("Tocando oreja derecha con ambas manos")
    
    # Tocarse la oreja izquierda
    if is_near(keypoints[LEFT_WRIST], keypoints[LEFT_EAR]):
        poses.append("Tocando oreja izquierda con mano izquierda")
    if is_near(keypoints[RIGHT_WRIST], keypoints[LEFT_EAR]):
        poses.append("Tocando oreja izquierda con mano derecha")
    if "Tocando oreja izquierda con mano izquierda" in poses and "Tocando oreja izquierda con mano derecha" in poses:
        poses.append("Tocando oreja izquierda con ambas manos")
    
    # Tocarse el hombro izquierdo
    if is_near(keypoints[LEFT_WRIST], keypoints[LEFT_SHOULDER]):
        poses.append("Tocando hombro izquierdo con mano izquierda")
    if is_near(keypoints[RIGHT_WRIST], keypoints[LEFT_SHOULDER]):
        poses.append("Tocando hombro izquierdo con mano derecha")
    
    # Tocarse el hombro derecho
    if is_near(keypoints[RIGHT_WRIST], keypoints[RIGHT_SHOULDER]):
        poses.append("Tocando hombro derecho con mano derecha")
    if is_near(keypoints[LEFT_WRIST], keypoints[RIGHT_SHOULDER]):
        poses.append("Tocando hombro derecho con mano izquierda")
    
    # Tocarse el codo
    if is_near(keypoints[RIGHT_WRIST], keypoints[LEFT_ELBOW]):
        poses.append("Tocando codo izquierdo con mano derecha")
    if is_near(keypoints[LEFT_WRIST], keypoints[RIGHT_ELBOW]):
        poses.append("Tocando codo derecho con mano izquierda")
    
    # Gestos de saludo (mano levantada y doblada cerca de la cabeza)
    if (is_above(keypoints[RIGHT_WRIST], keypoints[RIGHT_SHOULDER]) and 
        is_above(keypoints[RIGHT_WRIST], keypoints[RIGHT_ELBOW]) and
        is_near(keypoints[RIGHT_WRIST], keypoints[RIGHT_EAR], 100)):
        poses.append("Saludo con mano derecha")
    
    if (is_above(keypoints[LEFT_WRIST], keypoints[LEFT_SHOULDER]) and 
        is_above(keypoints[LEFT_WRIST], keypoints[LEFT_ELBOW]) and
        is_near(keypoints[LEFT_WRIST], keypoints[LEFT_EAR], 100)):
        poses.append("Saludo con mano izquierda")
    
    return poses

# Inicializar YOLO
model = YOLO("../models/yolov8m-pose.pt")

# Inicializar la cámara
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

cv2.namedWindow("Detector de Poses", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Procesar el frame con YOLO
    results = model(frame)
    annotated_frame = results[0].plot()

    # Obtener keypoints
    keypoints = results[0].keypoints
    if keypoints is not None and len(keypoints.xy) > 0:
        # Convertir keypoints a lista de puntos (x, y)
        person_keypoints = []
        for j, (x, y) in enumerate(keypoints.xy[0].cpu().numpy()):
            person_keypoints.append((float(x), float(y)))
        
        # Detectar poses
        poses = detect_poses(person_keypoints)
        
        # Mostrar poses detectadas en el frame
        y_pos = 30
        for pose in poses:
            cv2.putText(annotated_frame, pose, (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_pos += 30
    
    # Voltear la imagen horizontalmente para efecto espejo
    # annotated_frame = cv2.flip(annotated_frame, 1)
    
    # Mostrar frame
    cv2.imshow("Detector de Poses", annotated_frame)
    
    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
