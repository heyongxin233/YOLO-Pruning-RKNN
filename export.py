from ultralytics import YOLO
model = YOLO('yolo11.pt')
model.export(format='rknn')