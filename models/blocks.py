
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F



# Currently not in use
class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, X):
        X_rehsaped = torch.reshape(X, (self.shape, -1))
        return X_rehsaped