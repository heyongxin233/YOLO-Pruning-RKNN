# YOLO-Pruning-RKNN
Easy Training Official YOLOv8、YOLOv7、YOLOv6、YOLOv5、RT-DETR、Prune all_model using Torch-Pruning and Export RKNN Supported models!

We implemented YOLOv7 anchor free like YOLOv8！

We replaced the YOLOv8's operations that are not supported by the rknn NPU with operations that can be loaded on the NPU, all **without altering the original structure of YOLOv8**.

We implemented pruning of the YOLO model using **torch-pruning**.

You can reduce the number of parameters by **75%** without losing any accuracy!

New parameters:
```
prune: False(default):(bool) Whether to use torch-pruning 
prune_ratio: 0.66874(default):(float) Expected model pruning rate
prune_iterative_steps: 1(default):(int) Number of iteration rounds of pruning
prune_load: False(default):(bool) Whether to load weights after pruning
```
New model:
```
yolov7.yaml
```

You can use this code like [ultralytics for yolov8](https://github.com/ultralytics/ultralytics) ，and see the [YOLOv8 Docs](https://docs.ultralytics.com/) for full documentation on training, validation, prediction and deployment.
### Quickstart
```
pip install torch-pruning 
pip install -r requirements.txt
```
### Train and prune

**training example for yolov7**

You can see train.py
```
from ultralytics import YOLO
model = YOLO('yolov7m.yaml')
results = model.train(data='coco.yaml', epochs=100, imgsz=640, batch=64, device=[0,1,2,3],name='yolov7')
```
**pruning example for yolov8m**

You can see prune.py
```
from ultralytics import YOLO
model = YOLO('yolov8m.yaml')
results = model.train(data='coco.yaml', epochs=100, imgsz=640, batch=64, device=[0,1,2,3],name='yolov8_pruning',\
                      prune=True,prune_ratio=0.66874,prune_iterative_steps=1)
```

### Export

**export example for rknn**

You can see export.py,We support exporting the model to onnx supported by rknn npu.
```
from ultralytics import YOLO
model = YOLO('./yolov8m.pt')
model.export(format='rknn')
```

### Predict

You can predict model like ultralytics.You can see infer.py.More details see the [Predict](https://docs.ultralytics.com/modes/predict/) page
```
from ultralytics import YOLO
model = YOLO('yolov8n.pt') # model = YOLO('prune.pt')
model.predict('ultralytics/assets/bus.jpg',save=True,device=[0],line_width=2)
```

### Calculate model parameters
```
pip install thop
```
You can calculate model parameters and flops by using calculate.py
