import torch

print("============== Test dnntest.py ==============")
tensor = torch.randn((16,16)).cuda()
ret = torch.mm(tensor,tensor)
print(ret.cpu())
print("=============================================")