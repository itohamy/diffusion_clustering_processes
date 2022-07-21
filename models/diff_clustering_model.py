

import torch.nn as nn
import torch
import numpy as np
from models.feature_extraction import Encoder
from models.pre_processing import PreProcess

from . import utils, layers, layerspp, normalization
@utils.register_model(name='diff_clustering')
class DiffClustering(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config).to(self.device)
        self.preprocess = PreProcess(config).to(self.device)

    def forward(self, X, Z, t_labels):
        ''' X: input data, shape: [B, N, nc, img_sz, img_sz] where N is the same in each entry of the batch.
            Z: perturbed ground-truth continous representation of the discrete labels of X, shape: [B, N, K]
            t: std used to perturb Z, shape: [B]
        '''

        # Feature extractor:
        X_enc = self.encoder(X)  # shape: [B, N, d]

        # pre-process: input: [B, N, d], output: [B, N, H]
        Q = self.preprocess(X_enc, Z)

        # Score network: input: output:


        return

