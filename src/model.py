import torch
import torch.nn as nn

class Word2Vec(nn.Module):
    def __init__(self, word_size, hidden_size, w2i, i2w):
        super().__init__()
        self.A = nn.Linear(word_size, hidden_size)
        self.B = nn.Linear(hidden_size, word_size)
        self.sm = nn.SoftMax()
    
    def forward(self, X):
        x = torch.tensor([w2i[X[i]] for i in X])

        x = self.A(x)
        x = self.B(x)

        return self.sm(x)




