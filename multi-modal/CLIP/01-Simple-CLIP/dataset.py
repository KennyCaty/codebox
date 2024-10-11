from torch.utils.data import Dataset
from torchvision.transforms.v2 import PILToTensor, Compose
import torchvision

# 手写数字数据集
class MNIST(Dataset):
    def __init__(self, is_train=True):
        super().__init__()
        self.ds = torchvision.datasets.MNIST('./datasets/mnist/', train=is_train, download=True)
        self.img_convert = Compose([ 
            PILToTensor(),   
        ])
    
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        img, label = self.ds[index]
        return self.img_convert(img)/255.0, label
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    ds = MNIST(is_train=False)
    img, label = ds[0]
    print(label) 
    print("shape", img.shape)
    plt.imshow(img.permute(1, 2, 0)) # 将 PyTorch 张量的维度重新排列为 Matplotlib 能够理解的格式。
    """ 
    torch 的图片格式 [C, H, W]
    matplot 的图片格式 [H, W, C]
    """
    plt.show()