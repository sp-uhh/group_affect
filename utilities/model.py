import torch
import torch.nn as nn

from models import MLP


def build_mlp(config):
    model = MLP(config)
    return model