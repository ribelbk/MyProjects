from ultralytics import YOLO
import cv2

# Charger le modèle YOLOv8
model = YOLO("runs/detect/train8/weights/best.pt")

# Charger une image
image_path = "test.jpg"  # Remplace par le chemin de ton image
image = cv2.imread(image_path)

# Vérifier si l'image est bien chargée
if image is None:
    print("Erreur : Impossible de charger l'image. Vérifie le chemin.")
else:
    # Faire la prédiction
    results = model(image)

    # Dessiner les bounding boxes manuellement
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordonnées (coin supérieur gauche et inférieur droit)
            confidence = box.conf[0]  # Score de confiance
            label = f"{result.names[int(box.cls[0])]}: {confidence:.2f}"

            # Dessiner le rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Vert
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Afficher l'image
    cv2.imshow("Détection YOLOv8", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
