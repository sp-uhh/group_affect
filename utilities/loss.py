import torch
from torchmetrics.regression import ConcordanceCorrCoef

def CCCLoss(x, y):
    
        # x = torch.tensor(x)
        # y = torch.tensor(y)
        
        # concordance = ConcordanceCorrCoef()
        # ccc = concordance(y, x)
        
        mean1 = torch.mean(x)
        mean2 = torch.mean(y)
        std1 = torch.std(x)
        std2 = torch.std(y)
        dm = mean1 - mean2

        ccc = (
                (2 * (torch.mean((x - mean1) * (y - mean2)))) /
                ((std1 * std1) + (std2 * std2) + (dm * dm))
                )

        return ccc