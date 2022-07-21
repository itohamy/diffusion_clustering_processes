import torch.nn as nn
import torchvision.models as models
import torch


class Encoder(nn.Module):
    def __init__(self, config):

        super(Encoder, self).__init__()
        encoder_type = config.encoder.type  # options: {'resnet18', 'resnet50'}
        self.enc_output_dim = config.encoder.output_dim  # this is the reduced dimension.
        self.nc = config.data.num_channels
        self.img_sz = config.data.image_size

        self.model_dict = {"resnet18": models.resnet18(pretrained=False, num_classes=self.enc_output_dim),
                            "resnet50": models.resnet50(pretrained=False, num_classes=self.enc_output_dim)}

        self.backbone = self._get_basemodel(encoder_type)
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name):
        ''' model_name can be: {'resnet18', 'resnet50'} '''

        model = self.model_dict[model_name]
        return model


    def forward(self, X):
        ''' X shape: (B, N, nc, img_sz, img_sz) '''

        B = X.size(0)
        N = X.size(1)
        d = self.enc_output_dim

        X_rehsaped = torch.reshape(X, (B * N, X.size(2), X.size(3), X.size(4)))  # reshape to [B * N, nc, img_sz, img_sz]

        X_rehsaped_emb = self.backbone(X_rehsaped)  # (B * N, d=enc_output_dim)

        X_emb = torch.reshape(X_rehsaped_emb, (B, N, d))   # reshape to [B, N, d]

        return X_emb

