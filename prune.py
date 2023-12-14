from ultralytics import YOLO
model = YOLO('yolov7m.yaml')
results = model.train(data='coco.yaml', epochs=10, imgsz=1280, batch=8, device=[3],name='yolov7',prune=True,prune_ratio=0.66874,prune_iterative_steps=1)
