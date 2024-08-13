import torch
from torchvision.models.vision_transformer import vit_b_16
from torchvision.models.vision_transformer import ViT_B_16_Weights

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT).to(device)
inputs = torch.randn(1,3,224,224).to(device)
with torch.inference_mode():
    y = model(inputs)