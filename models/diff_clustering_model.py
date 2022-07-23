

import torch.nn as nn
import torch
import numpy as np
from models.feature_extraction import Encoder
from models.pre_processing import PreProcess
from models.score_network import ScoreNetwork


from . import utils, layers, layerspp, normalization
@utils.register_model(name='diff_clustering')
class DiffClustering(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config).to(self.device)
        self.preprocess = PreProcess(config).to(self.device) # TBD
        self.score_net = ScoreNetwork(config).to(self.device) # TBD
        self.fc1 = None # TBD
        self.fc2 = None # TBD
        self.fc3 = None # TBD
        self.fc4 = None # TBD

    def forward(self, X, Z_t, t_labels):
        ''' X: input data, shape: [B, N, nc, img_sz, img_sz] where N is the same in each entry of the batch.
            Z_t: perturbed ground-truth continous representation of the discrete labels of X, shape: [B, N, K]
            t: std used to perturb Z, shape: [B]
        '''

        # Feature extractor:
        X_enc = self.encoder(X)  # shape: [B, N, d]

        # Expand Z_t representation:
        Z_t = self.fc1(Z_t)  # shape: [B, N, K, q]

        # pre-process: Input: X: [B, N, d] and Z_t: [B, N, K, q]. Output: [B, N, K, d+q]
        Q = self.preprocess(X_enc, Z_t)

        # Expand Q representation:
        Q = self.fc2(Q)  # shape: [B, N, K, H]

        # prepare t:
        t = self.fc3(t_labels)
        t = self.fc4(t)

        # Score network: input: Q: [B, N, K, H] and t: [B, H]. Output: [B, N, K]
        score = self.score_net(Q, t)

        return score

