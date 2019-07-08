import torch as th
import torch.nn as nn

class GATLayer(nn.Module):
    def __init__(self, w_user, b_user, w_item, b_item,
                 user_att_heads, item_att_heads, nonlinear, merge):
        super().__init__()
        self.w_user = w_user
        self.b_user = b_user
        self.w_item = w_item
        self.b_item = b_item
        self.user_att_heads = user_att_heads
        self.item_att_heads = item_att_heads
        self.nonlinear = nonlinear
        self.merge = merge

    @staticmethod
    def neighborhood_softmax(adj, att):
        pass

    def attend(self, adj, h_src, h_dst, att_heads):
        zs = []
        for att_head in att_heads:
            logit = att_head(h_src, th.repeat_interleave(h_dst, adj.degree, 0))
            att = self.neighborhood_softmax(adj, logit)
            z = self.nonlinear(ths.FloatTensor(adj.idx, att, adj.size) @ h_src)
            zs.append(z)
        return self.merge(zs)

    def forward(self, u2i, i2u, h_user, h_item):
        h_user = h_user @ self.w_user + self.b_user
        h_item = h_item @ self.w_item + self.b_item
        h_user = attend(self, u2i, h_user, h_item, self.user_att_heads)
        h_item = attend(self, u2i, h_item, h_user, self.item_att_heads)
        return h_user, h_item

class Attention(nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        self.a_src = nn.Parameter(th.rand(in_feats))
        self.a_dst = nn.Parameter(th.rand(in_feats))
        self.b = 0
#       self.b = nn.Parameter(th.zeros(1))

    def forward(self, h_src, h_dst):
        return F.leaky_relu(h_src @ self.a_src + h_dst @ self.a_dst + self.b)

class GAT(nn.Module):
    def __init__(self, n_feats, n_att_heads, nonlinear):
        super().__init__()
        layers = nn.ModuleList()
        for i, [in_feats, out_feats] in enumerate(zip(n_feats[:-1], n_feats[1:])):
            w_user = nn.Parameter(th.rand(in_feats, out_feats))
            b_user = 0
#           b_user = nn.Parameter(th.zeros(1, out_feats))
            w_item = nn.Parameter(th.rand(in_feats, out_feats))
            b_item = 0
#           b_item = nn.Parameter(th.zeros(1, out_feats))
            user_att_heads = nn.ModuleList([Attention(out_feats) for i in range(n_att_heads)])
            item_att_heads = nn.ModuleList([Attention(out_feats) for i in range(n_att_heads)])
            concatenate = lambda xs: th.cat(xs, 1)
            average = lambda xs: sum(xs) / len(xs)
            merge = concatenate if i < len(n_feats) - 1 else average
            layers.append(GATLyaer(w_user, b_user, w_item, b_item,
                                   user_att_heads, item_att_heads, nonlinear, merge))

    def forward(self, u2i, i2u, x_user, x_item):
        h_user = x_user
        h_item = x_item
        for layer in self.layers:
            h_user, h_item = layer(h_user, h_item)
        return h_user, h_item
