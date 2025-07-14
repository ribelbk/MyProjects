import depthai as dai
from ultralytics import YOLO
import cv2
import numpy as np
import time  # Importer time pour le délai

# Charger le modèle YOLOv8
model = YOLO("runs/detect/train8/weights/best.pt")  # Mets le bon chemin vers ton modèle

# Initialiser le pipeline OAK-D Lite
pipeline = dai.Pipeline()

# Créer un noeud pour la caméra
cam = pipeline.create(dai.node.ColorCamera)
cam.setPreviewSize(640, 640)
cam.setInterleaved(False)
cam.setBoardSocket(dai.CameraBoardSocket.RGB)
cam.setFps(30)  # Fixer le FPS à 30 pour une meilleure fluidité

# Création d'un output pour récupérer les frames
xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("video")
cam.preview.link(xout.input)

# Lancer l'OAK-D Lite
with dai.Device(pipeline) as device:
    queue = device.getOutputQueue(name="video", maxSize=8, blocking=False)

    time.sleep(2)  # ✅ Ajouter un délai de 2 secondes pour éviter les erreurs de démarrage

    while True:
        imgFrame = queue.get()
        frame = imgFrame.getCvFrame()  # Récupérer l'image

        # Faire la détection avec YOLO
        results = model(frame)

        # Dessiner les bounding boxes
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordonnées de la boîte
                confidence = box.conf[0]  # Score de confiance
                label = f"{result.names[int(box.cls[0])]}: {confidence:.2f}"

                # Dessiner le rectangle autour de l'objet détecté
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Vert
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Afficher la vidéo en direct avec les détections
        cv2.imshow("YOLOv8 OAK-D Lite", frame)

        # Quitter avec la touche 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Fermer les fenêtres
cv2.destroyAllWindows()
