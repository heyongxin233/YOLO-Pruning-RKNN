# 🚀 YOLO-Pruning-RKNN 🚀

## ✨ Key Features
- **Super-Efficient Training**: Train state-of-the-art object detection models as easily as using official Ultralytics YOLO ⚡
- **Smart Model Pruning**: Use Torch-Pruning to prune models, reducing parameters by up to ​75% without losing accuracy 🎯
- **RKNN Export Support**: One-click export to RKNN format for seamless deployment on Rockchip NPU platforms 🚀
- **Full-Toolchain Support**: End-to-end workflow covering training, validation, inference, and deployment 🛠️

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

#### Quick Purning

Directly pruning and training pre-trained models generally results in lower accuracy and larger models. See `prune.py` for details.

```python
prunetrain(quick_pruning=True,        # Quick Pruning or not
           data='coco.yaml',          # Dataset config
           train_epochs=10,           # Epochs before pruning
           imgsz=640,                 # Input size
           batch=8,                   # Batch size
           device=[0],                # GPU devices
           name='yolo11',             # Save name
           prune_ratio=0.5,           # Pruning Ratio (50%)
           prune_iterative_steps=1    # Pruning Interative Steps
)
```

#### Normal Purning

According to the author of torch pruning, first train, then prune, and retrain after pruning.

In normal pruning mode, the `prune_epochs` parameter is mandatory, representing the number of training epochs after pruning. 

See `prune.py` for details.

```python
prunetrain(quick_pruning=False,       # Quick Pruning or not
           data='coco.yaml',          # Dataset config
           train_epochs=10,           # Epochs before pruning
           prune_epochs=10,           # Epochs after pruning 
           imgsz=640,                 # Input size
           batch=8,                   # Batch size
           device=[0],                # GPU devices
           name='yolo11',             # Save name
           prune_ratio=0.5,           # Pruning Ratio (50%)
           prune_iterative_steps=1,   # Pruning Interative Steps
           sparse_training=False      # Experimental, Allow Sparse Training Before Pruning
)
```

Please note that the `sparse_training` parameter is experimental, and setting it to `True` may result in **better or worse** performance.

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

## 📚 Acknowledgements

- Special thanks to [@VainF](https://github.com/VainF) for their contribution to the [Torch-Pruning](https://github.com/VainF/Torch-Pruning) project! This project relies on it for model pruning.

- Special thanks to [@Ultralytics](https://github.com/ultralytics) for their contribution to the [ultralytics](https://github.com/ultralytics/ultralytics) project! This project relies on it for the framework.