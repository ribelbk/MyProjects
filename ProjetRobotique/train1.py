from ultralytics import YOLO

# Charger YOLOv8 pré-entraîné
model = YOLO("yolov8n.pt")

# Lancer l'entraînement
model.train(data="C:/Users/bakho/Documents/ProjetRob/can_dataset/dataset.yaml", epochs=10, imgsz=640, batch=16)

