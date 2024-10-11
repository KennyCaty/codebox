from torch import nn
import torch
import torch.nn.functional as F

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 为了简化，把Mnist数据集的label当作text，就只有0-9，所以不需要Transformer这样复杂的语言理解与Encode能力，直接做Embedding和mlp层即可
        self.emb = nn.Embedding(num_embeddings=10, embedding_dim=16)
        self.dense1 = nn.Linear(16, 64)
        self.dense2 = nn.Linear(64, 16)
        self.wt = nn.Linear(16, 8)  # 同一特征长度 8 
        self.ln = nn.LayerNorm(8)
        
    def forward(self, x):
        x = self.emb(x)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.wt(x)
        x = self.ln(x)
        return x
    
if __name__=="__main__":
    text_encoder = TextEncoder()
    x = torch.tensor([1,2,3,4,5,6,7,8,9,0])
    y = text_encoder(x)
    print(y.shape)