
import torch.nn as nn
import torchvision.models as models
import torch
from scipy.stats import truncnorm
import numpy as np


class PreProcess(nn.Module):

    def __init__(self, config):
        super(PreProcess, self).__init__()


    def forward(self, X, Z_t):
        '''
            X (input): input data, shape: [B, N, d] where N is the same in each entry of the batch and d is a small dimension of the data.
            Z_t (input): perturbed ground-truth continous representation of the discrete labels of X, shape: [B, N, K, q]
            Q (output): combnation of Z and X, shape: [B, N, K, d+q]
        '''


        Q = ..

        return Q

        



