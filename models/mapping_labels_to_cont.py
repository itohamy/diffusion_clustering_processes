
import torch.nn as nn
import torchvision.models as models
import torch
from scipy.stats import truncnorm
import numpy as np


class MappingToContinuous(nn.Module):

    def __init__(self, config):
        super(MappingToContinuous, self).__init__()
        self.K = config.data.nlabels
        self.mu = config.mapping_to_cont.mu
        self.sigma = config.mapping_to_cont.sigma


    def forward(self, C):
        '''
            C (input): ground-truth labels of X, shape: [B, N]
            Z (output): continouos version of C, shape: [B, N, K]
        '''

        B, N = C.size()
        Z = torch.ones((B, N, self.K))
        
        for b in range(B):
            C_b = C[b]  # [N,]
            
            # Sample from truncated normal distribution
            upper_bnds = np.random.normal(self.mu, self.sigma, N)  # these are the z_{i,c_i} that represents the highest value in every row in Z[b]. (in the cell: i, c_i)

            for i in range(N):
                z_i = truncnorm.rvs(float('-inf'), upper_bnds[i], size=self.K)
                z_i[C_b[i]] = upper_bnds[i]
                Z[b, i, :] = z_i

        # Dense layer to create [B, N, K] --> [B, N, K, q]:
        # ....

        return Z

        



