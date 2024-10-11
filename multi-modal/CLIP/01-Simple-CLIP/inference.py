"""
CLIP图像分类功能
"""
from dataset import MNIST
import matplotlib.pyplot as plt
import torch
from clip import CLIP
import torch.nn.functional as F


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dataset = MNIST(is_train=False)

model = CLIP().to(DEVICE)
model.load_state_dict(torch.load("./checkpoints/clip/simple_clip_model.pth"))

model.eval() 

"""
分类
"""

images, labels = dataset[0]
print("正确的类别：", labels)
plt.imshow(images.permute(1, 2, 0))
plt.show()

targets = torch.arange(0, 10)
logits = model(images.unsqueeze(0).to(DEVICE), targets.to(DEVICE))  # 10张图片 vs 10种分类
print(logits)
print("CLIP分类：", logits.argmax(-1).item())
