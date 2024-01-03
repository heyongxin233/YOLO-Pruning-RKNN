from ultralytics import YOLO
from thop import profile
import torch
input1 = torch.randn(1, 3, 480, 640) 
model = YOLO('/opt/yolov8/runs/detect/yolov8s-leakey-relu-prune3/weights/best.pt')
flops, params = profile(model.model, inputs=(input1, ))
print('FLOPs = ' + str(flops/1000**3) + 'G')
print('Params = ' + str(params/1000**2) + 'M')