import torch.nn as nn

class CollapsedMF(nn.Module):
    def __init__(self, mf):
        super().__init__()
        self.mf = mf

    def forward(self, uu, ii, rr, v, j, mm, s=None):
        return self.mf(None, None, r, v, j, s=None), None

class TemporalMF(nn.Module):
    def __init__(self, mf, merge, T):
        super().__init__()
        self.mf = mf
        self.merge = merge
        self.T = T

    def forward(self, uu, ii, rr, v, j, mm, s=None):
        pass
