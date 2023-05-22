import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchMixLoss(nn.Module):
    """ Refer to models/ast.py for the code of patch mixing
    """
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion

    def forward(self, pred, y_a, y_b, lam):
        loss = lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)
        return loss
