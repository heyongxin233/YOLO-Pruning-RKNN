from ultralytics import YOLO
model = YOLO('yolov8m.yaml')
results = model.train(data='coco.yaml', epochs=10, imgsz=1280, batch=8, device=[3],name='yolov8m',prune=False)
