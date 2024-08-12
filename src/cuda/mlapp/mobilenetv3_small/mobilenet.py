import torch
from torchvision.models.mobilenetv3 import mobilenet_v3_small 
from torchvision.models.mobilenetv3 import MobileNet_V3_Small_Weights

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).to(device)
inputs = torch.randn(1,3,224,224).to(device)
y = model(inputs)