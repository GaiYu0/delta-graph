import torch.nn as nn

class GATLayer(nn.Module):
    def __init__(self, in_feats, out_feats, attention_heads):
        super().__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.attention_heads = attention_heads

    def forward(self, g):
        g.apply_nodes(lambda nodes: {'h' : self.linear(nodes.data['h'])})
