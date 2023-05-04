import torch
from torchvision import transforms
import numpy as np
# 定义transforms
trans = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
])

# 假设color_map是1 1280 1280 3的张量
alpha_map = torch.randn(160, 160)
alpha_map_expanded = np.zeros((160, 160, 3))
alpha_map_expanded[:,:,0] = alpha_map
alpha_map_expanded[:,:,1] = alpha_map
alpha_map_expanded[:,:,2] = alpha_map

# 打印输出张量的形状
print(alpha_map_expanded.shape)
