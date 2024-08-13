import torch
from torchvision.models.resnet import resnet50
from torchvision.models.resnet import ResNet50_Weights

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
inputs = torch.randn(1,3,224,224).to(device)
with torch.inference_mode():
    y = model(inputs)