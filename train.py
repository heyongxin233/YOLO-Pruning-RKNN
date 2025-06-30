from ultralytics import YOLO
model = YOLO('yolo11.yaml')
results = model.train(data='coco.yaml', epochs=10, imgsz=640, batch=8, device=[0], name='yolo11', prune=False)
