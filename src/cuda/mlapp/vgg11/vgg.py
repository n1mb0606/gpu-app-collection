import torch
from torchvision.models.vgg import vgg11
from torchvision.models.vgg import VGG11_Weights

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = vgg11(weights=VGG11_Weights.DEFAULT).to(device)
inputs = torch.randn(1,3,224,224).to(device)
with torch.inference_mode():
    y = model(inputs)