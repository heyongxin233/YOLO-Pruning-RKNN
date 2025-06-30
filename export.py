from ultralytics import YOLO
model = YOLO('yolov11.pt')
model.export(format='rknn')