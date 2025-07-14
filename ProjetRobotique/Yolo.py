from ultralytics import YOLO
import cv2

# Charger le modèle YOLOv8 Nano
model = YOLO("yolov8n.pt")

# Charger l'image
image_path = "test.jpg"  # Remplace par ton image
image = cv2.imread(image_path)

# Faire la détection
results = model.predict(image, conf=0.5)

# Dessiner les boîtes autour des objets détectés
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Afficher l'image avec détection
cv2.imshow("YOLOv8 - Image Test", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
