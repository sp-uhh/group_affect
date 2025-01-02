import torch.nn as nn

class MLP(nn.Module):
    
    def __init__(self, config):
        super(MLP, self).__init__()

        self.dense1 = nn.Linear(config.nfeatures, 128)
        self.act1   = nn.ReLU(inplace=True)
        self.norm1  = nn.BatchNorm1d(128, affine=True)
        
        self.dense2 = nn.Linear(128, 64)
        self.act2    = nn.ReLU(inplace=True)
        self.norm2  = nn.BatchNorm1d(64, affine=True)
        
        self.out = nn.Linear(64, config.nlabels)
        
    def forward(self, x):
        
        x = self.dense1(x)
        x = self.act1(x)
        x = self.norm1(x)
        
        x = self.dense2(x)
        x = self.act2(x)
        x = self.norm2(x)
        
        x = self.out(x)
        
        return x