from ultralytics import YOLO
model = YOLO('/opt/yolov8/runs/detect/prune2/weights/best.pt')
model.export(format='onnx')