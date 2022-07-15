

import torch.nn as nn
import torch
import numpy as np


from . import utils, layers, layerspp, normalization
@utils.register_model(name='diff_clustering')
class DiffClustering(nn.Module):
    
    def __init__(self, config):
        super().__init__()

    def forward(self, x):

        # feature extractor: input: [B, N, D], output: [B, N, d]

        # pre-process: input: [B, N, d], output: [B, N, H]

        # main network: input: output:


        return

