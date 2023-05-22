import torch.nn as nn


class Projector(nn.Module):
    def __init__(self, in_dim, out_dim=768, apply_bn=True):
        super(Projector, self).__init__()
        self.linear1 = nn.Linear(in_dim, in_dim)
        self.linear2 = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(in_dim)
        self.relu = nn.ReLU()
        if apply_bn:
            self.projector = nn.Sequential(self.linear1, self.bn, self.relu, self.linear2)
        else:
            self.projector = nn.Sequential(self.linear1, self.relu, self.linear2)

    def forward(self, x):
        return self.projector(x)