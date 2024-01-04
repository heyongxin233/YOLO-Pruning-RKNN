from ultralytics import YOLO
from thop import profile
import torch
from thop import clever_format
input1 = torch.randn(1, 3, 480, 640) 
model = YOLO('./yolov8n.pt')
macs, params = profile(model.model, inputs=(input1, ))
macs, params = clever_format([macs, params], "%.3f")
print('macs :',macs, 'params:',params)