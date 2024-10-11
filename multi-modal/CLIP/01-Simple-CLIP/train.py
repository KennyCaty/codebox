import torch
from dataset import MNIST
from clip import CLIP
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

dataset = MNIST() # 加载数据集

model = CLIP().to(DEVICE)  # 加载模型

try: # 加载权重
    model.load_state_dict(torch.load('./checkpoints/clip/simple_clip_model.pth'))
except:
    pass

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

'''
    训练  
'''
EPOCH = 100000
BATCH_SIZE = 64
TARGET_COUNT = 10 #共10种数字

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

for i in range(EPOCH):
    while True:  ## 必须选出10种不一样的数字 实际上每一个batch只有10个不同数字
        imgs, labels = next(iter(dataloader))
        if torch.unique(labels).shape[0]<TARGET_COUNT: # 未覆盖10种数字
            continue
        target = set()
        indexes = []
        for j in range(BATCH_SIZE):
            if labels[j].item() in target:
                continue
            target.add(labels[j].item())
            indexes.append(j)
            if len(target) == TARGET_COUNT:
                break
        imgs = imgs[indexes]
        labels = labels[indexes]
        break
    
    logits = model(imgs.to(DEVICE), labels.to(DEVICE))
    
    targets = torch.arange(0, TARGET_COUNT).to(DEVICE)
    
    loss_i = F.cross_entropy(logits, targets)
    loss_t = F.cross_entropy(logits.permute(1, 0), targets)
    loss = (loss_i + loss_t)/2
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i%1000==0:
        print("iter:{}, loss:{}".format(i, loss))
        torch.save(model.state_dict(), './checkpoints/clip/.simple_clip_model.pth')  # 先保存为临时文件
        os.replace('./checkpoints/clip/.simple_clip_model.pth', './checkpoints/clip/simple_clip_model.pth') #替换为正式文件（防止写入损坏，提高稳定性）