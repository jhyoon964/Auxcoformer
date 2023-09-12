from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('')

model.predict('', save = True, show=False, imgsz=640, conf=0.5)
