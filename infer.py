from ultralytics import YOLO
model = YOLO('yolov8n.pt') # model = YOLO('prune.pt')
model.predict('ultralytics/assets/bus.jpg',save=True,device=[0],line_width=2)