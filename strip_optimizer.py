from ultralytics.utils.torch_utils import strip_optimizer
f='/opt/yolov8/runs/detect/new_yplov8m/weights/best.pt'
strip_optimizer(f)