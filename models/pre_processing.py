
import torch.nn as nn
import torchvision.models as models
import torch
from scipy.stats import truncnorm
import numpy as np


class PreProcess(nn.Module):

    def __init__(self, config):
        super(PreProcess, self).__init__()

    def forward(self, X, Zt):
        '''
            X (input): input data, shape: [B, N, d] where N is the same in each entry of the batch and d is a small dimension of the data.
            Zt (input): perturbed ground-truth continous representation of the discrete labels of X, shape: [B, N, K, q]
            ZcondX (output): combination of Z and X, shape: [B, N, K, d+q]
        '''
        
        B, N, K, q = Zt.size()

        X_ = X[:, :, None, :]  # shape: [B, N, 1, d]
        X__ = X_.expand(-1, -1, K, -1)  # shape: [B, N, K, d]

        ZcondX = torch.cat((Zt, X__), dim=3)  # shape: [B, N, K, d + q]

        return ZcondX

        



