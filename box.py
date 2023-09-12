from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('cc/cc/weights/best.pt')

model.predict('D:/Crack/Crack/concreteCrackSegmentationDataset/cc_weather/val_rain', save = True, show=False, imgsz=448, conf=0.5)