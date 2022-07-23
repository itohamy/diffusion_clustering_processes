

import torch.nn as nn
import torch
import numpy as np
from models.feature_extraction import Encoder
from models.pre_processing import PreProcess
from models.score_network import ScoreNetwork
from models import blocks


from . import utils, layers, layerspp, normalization
@utils.register_model(name='diff_clustering')
class DiffClustering(nn.Module):
    
    def __init__(self, config):
        super().__init__()

        self.d = config.encoder.output_dim  # Dimension of the data after feature extraction
        self.q = config.model.q  # Z expanded dim before pre-processing
        self.H = config.model.H  # Z|X expanded dim after pre-processing
        self.nf = config.model.nf

        self.encoder = Encoder(config).to(self.device)

        self.fc1_Z_expansion = nn.Sequential(nn.Linear(1, self.q), nn.BatchNorm1d(self.q), nn.LeakyReLU(0.2, inplace=True))
        self.preprocess = PreProcess(config).to(self.device)

        self.fc2_ZcondX_expansion = nn.Sequential(nn.Linear(self.d + self.q, self.H), nn.BatchNorm1d(self.H), nn.LeakyReLU(0.2, inplace=True))
        self.used_sigmas_embedding = layerspp.GaussianFourierProjection(embedding_size=self.nf, scale=config.model.fourier_scale)
        self.fc3_t_expansion = nn.Sequential(nn.Linear(self.nf * 2, self.H), nn.BatchNorm1d(self.H), nn.LeakyReLU(0.2, inplace=True))

        self.score_net = ScoreNetwork(config).to(self.device) # TBD

    def forward(self, X, Zt, t_labels):
        ''' X: input data, shape: [B, N, nc, img_sz, img_sz] where N is the same in each entry of the batch.
            Zt: perturbed ground-truth continous representation of the discrete labels of X, shape: [B, N, K]
            t_labels: std used to perturb Z, shape: [B]
        '''

        B, N, K = Zt.size()

        # Feature extractor:
        X_enc = self.encoder(X)  # shape: [B, N, d]

        # Expand Zt representation (1 --> q):
        Zt = torch.reshape(Zt, (B * N * K, 1))
        Zt = self.fc1_Z_expansion(Zt)  # shape: [B * N * K, q]
        Zt = torch.reshape(Zt, (B, N, K, self.q)) # shape: [B, N, K, q]

        # Pre-process: Input: X: [B, N, d] and Zt: [B, N, K, q]. Output: ZcondX: [B, N, K, d+q]
        ZcondX = self.preprocess(X_enc, Zt)

        # Expand ZcondX representation (d+q --> H):
        ZcondX = torch.reshape(ZcondX, (B * N * K, self.d + self.q))
        ZcondX = self.fc2_ZcondX_expansion(ZcondX)  # shape: [B, N, K, H]
        ZcondX = torch.reshape(ZcondX, (B, N, K, self.H)) # shape: [B, N, K, H]

        # prepare t:
        t = self.used_sigmas_embedding(torch.log(t_labels))  # shape: [B, 2 * nf]
        t = self.fc3_t_expansion(t)  # shape: [B, H]

        # Score network: input: ZcondX: [B, N, K, H] and t: [B, H]. Output: score: [B, N, K]
        score = self.score_net(ZcondX, t)  # shape: [B, N, K]
        
        return score

