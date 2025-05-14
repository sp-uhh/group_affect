import torch
import torch.nn as nn

from models import MLP, GAT


def build_mlp(config):
    model = MLP(config)
    return model

# Graph Neural Network
def build_gcn(config):
    model = GAT(config.nfeatures, config.hidden_channels, config.nlabels, heads=config.heads, batch_size=config.batch_size)
    return model