# YOLO-Pruning
Easy Training Official YOLOv8、YOLOv7、YOLOv6、YOLOv5  and Prune all_model using Torch-Pruning!

We implemented YOLOv7 anchor free like YOLOv5！

You can use this code like [ultralytics for yolov8](https://github.com/ultralytics/ultralytics) 
```
pip install torch-pruning 
pip install -r requirements.txt
```
We implemented pruning of the YOLO model using torch-pruning.

You can reduce the number of parameters by **75%** without losing any accuracy!

New parameters:
```
prune: False(default):(bool) Whether to use torch-pruning 
prune_ratio: 0.66874(default):(float) Expected model pruning rate
prune_iterative_steps: 1(default):(int) Number of iteration rounds of pruning
```
**training example for yolov7**
```
from ultralytics import YOLO
model = YOLO('yolov7m.yaml')
results = model.train(data='coco.yaml', epochs=100, imgsz=640, batch=64, device=[0,1,2,3],name='yolov7')
```
**pruning example for yolov8m**
```
from ultralytics import YOLO
model = YOLO('yolov8m.yaml')
results = model.train(data='coco.yaml', epochs=100, imgsz=640, batch=64, device=[0,1,2,3],name='yolov8_pruning',\
                      prune=True,prune_ratio=0.66874,prune_iterative_steps=1)
```
