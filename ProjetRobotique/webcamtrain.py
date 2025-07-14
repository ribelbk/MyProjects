from ultralytics import YOLO
import cv2

# Charger le modèle YOLOv8
model = YOLO("runs/detect/train8/weights/best.pt")  # Mets le bon chemin du modèle

# Ouvrir la webcam
cap = cv2.VideoCapture(0)  # 0 pour la webcam par défaut, 1 si une autre caméra est branchée

if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir la webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de capturer une image.")
        break

    # Faire la détection avec YOLO
    results = model(frame)

    # Dessiner les bounding boxes
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordonnées des boîtes
            confidence = box.conf[0]  # Score de confiance
            label = f"{result.names[int(box.cls[0])]}: {confidence:.2f}"

            # Dessiner le rectangle autour de l'objet détecté
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Vert
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Afficher la vidéo avec détection en live
    cv2.imshow("YOLOv8 - Détection en live", frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
