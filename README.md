# 🚀 YOLO-Pruning-RKNN 🚀

## ✨ Key Features
​- **Super-Efficient Training**: Train state-of-the-art object detection models as easily as using official Ultralytics YOLO ⚡
​- **Smart Model Pruning**: Use Torch-Pruning to prune models, reducing parameters by up to ​75% without losing accuracy 🎯
​- **RKNN Export Support**: One-click export to RKNN format for seamless deployment on Rockchip NPU platforms 🚀
​- **Full-Toolchain Support**: End-to-end workflow covering training, validation, inference, and deployment 🛠️

## Quickstart
### 🔧 Install Dependencies
```bash
pip install torch-pruning 
pip install -r requirements.txt
```
## 🚂 Training & Pruning
### 📊 YOLO11 Training Example
```python
from ultralytics import YOLO

# Create and train a model
model = YOLO('yolo11.yaml')
results = model.train(
    data='coco.yaml',         # Dataset config file
    epochs=100,               # Number of training epochs
    imgsz=640,                # Image size
    batch=16,                 # Batch size
    device=[0,1,2,3],         # Use GPUs 0-3
    name='yolo11'             # Experiment name
)
```
### ✂️ YOLO11 Pruning Example

```python
from ultralytics import YOLO

# Create and prune a model (parameters are customizable)
model = YOLO('yolo11.yaml')
results = model.train(
    data='coco.yaml',
    epochs=100,
    imgsz=640,
    batch=64,
    device=[0,1,2,3],
    name='yolov8_pruning',
    prune=True,               # Enable pruning
    prune_ratio=0.5,          # Pruning ratio (50%)
    prune_iterative_steps=1   # Iterative pruning steps
)
```

## 📤 Model Export

### Export to RKNN Format
```python
from ultralytics import YOLO

# Load a trained model
model = YOLO('yolo11.pt')

# Export to RKNN format
model.export(format='rknn')
```

### 🔮 Model Inference

More details about [predict](https://docs.ultralytics.com/modes/predict/).
```python
from ultralytics import YOLO

# Load a model (original or pruned)
model = YOLO('yolo11.pt')  # or model = YOLO('pruned.pt')

# Run inference
model.predict(
    'ultralytics/assets/bus.jpg',  # Input image path
    save=True,                     # Save results
    device=[0],                    # Use GPU 0
    line_width=2                   # Detection box line width
)
```

## 🔢 Model Analysis
Use `thop` to easily calculate model parameters and FLOPs:
```bash
pip install thop
```
You can calculate model parameters and flops by using calculate.py

## 🤝 Contributing & Support
Feel free to submit issues or pull requests on GitHub for questions or suggestions!
