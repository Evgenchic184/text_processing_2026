import torch
import torch.nn as nn

class Word2Vec(nn.Module):
    def __init__(self, word_size, hidden_size, w2i, i2w):
        super().__init__()
        self.A = nn.Linear(word_size, hidden_size)
        self.B = nn.Linear(hidden_size, word_size)
        self.sm = nn.Softmax()

        self.w2i = w2i
        self.i2w = i2w
    
    def forward(self, X):
        import torch.nn.functional as F
        x = torch.tensor([self.w2i[i] for i in X])
        x = F.one_hot(x, num_classes=len(self.w2i)).float()
        x = self.A(x)
        x = self.B(x)

        return self.sm(x)




