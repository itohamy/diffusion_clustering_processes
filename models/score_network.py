import torch.nn as nn
import torchvision.models as models
import torch


class ScoreNetwork(nn.Module):
    def __init__(self, config):

        super(ScoreNetwork, self).__init__()


    def forward(self, ZcondX, t):
        ''' ZcondX: [B, N, K, H]
            t: the embedded sigmas that were used to perturb Z. Shape: [B, H]. 
            score (output): should match the score function of p(Z(t)|X). Shape: [B, N, K]
        '''

        

        score = None  # shape: [B, N, K]

        return score



