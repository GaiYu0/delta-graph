import torch as th
import dgl.function as fn

def neighborhood_softmax(g, in_key, out_key):
    g.apply_nodes(lambda nodes: {'exp' : th.exp(nodes.data[in_key])})
    g.pull(g.nodes(), fn.copy_src('exp', 'msg'), fn.reducer.max('msg', 'max'))
    g.apply_nodes(lambda nodes: {'exp' : nodes.data['exp'] - nodes.data['max']})
    g.pull(g.nodes(), fn.copy_src('exp', 'msg'), fn.reducer.sum('msg', 'sum'))
    g.apply_edges(lambda edges: {out_key : edges.src['exp'] / edges.dst['sum']})
