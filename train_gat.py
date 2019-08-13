import argparse
from itertools import chain, starmap

import numpy as np
import numpy.random as npr
from tensorboardX import SummaryWriter
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from adj import Adj, SubAdj
import gat, gcmc

parser = argparse.ArgumentParser()
parser.add_argument('--bs-train', type=int)
parser.add_argument('--bs-infer', type=int)
parser.add_argument('--dh', type=int)
parser.add_argument('--dropout-rate', type=float)
parser.add_argument('--dropout-type', type=str)
parser.add_argument('--ds', type=str)
parser.add_argument('--gpu', type=int)
parser.add_argument('--in-feats', type=int)
parser.add_argument('--lr', type=float)
parser.add_argument('--model', type=str)
parser.add_argument('--n-att-heads', type=int)
parser.add_argument('--n-bases', type=int)
parser.add_argument('--n-epochs', type=int)
parser.add_argument('--n-feats', type=int, nargs='+')
parser.add_argument('--nonlinear', type=str)
parser.add_argument('--normalization', type=str)
parser.add_argument('--out-feats', type=int)
parser.add_argument('--p-train', type=float)
parser.add_argument('--p-val', type=float)
parser.add_argument('--reduce-chnl', type=str)
parser.add_argument('--x', action='store_true')
args = parser.parse_args()

uid = np.load(args.ds + '/uid.npy')
iid = np.load(args.ds + '/iid.npy')
idx = np.load(args.ds + '/idx.npy')
rating = np.load(args.ds + '/rating.npy')
uniq_rating = np.unique(rating)
masks = [(idx == i) for i in np.unique(idx)][:-1]
sub_uids = list(map(uid.__getitem__, masks))
sub_iids = list(map(iid.__getitem__, masks))
sub_ratings = list(map(rating.__getitem__, masks))

x_user = np.load(args.ds + '/x-user.npy')
x_item = np.load(args.ds + '/x-item.npy')

n_users = len(x_user)
n_items = len(x_item)
n_chnls = len(uniq_rating)

device = th.device('cpu') if args.gpu < 0 else th.device('cuda:%d' % args.gpu)
from_numpy = lambda x: th.from_numpy(x).to(device)
nonlinear = getattr(F, args.nonlinear)
if args.model == 'gat':
    embedding_user = nn.Embedding(n_users, args.dh).to(device)
    embedding_item = nn.Embedding(n_items, args.dh).to(device)
    args.n_feats.insert(0, args.dh + (x_item.size(1) if args.x else 0))
    gat = gat.MCGAT(args.n_feats[:-1], args.n_att_heads, n_chnls, args.reduce_chnl, nonlinear).to(device)
    d = 2 * args.n_feats[-2] * (1 if args.reduce_chnl == 'sum' else n_chnls)
    rp = gat.RatingPredictor(d, args.n_feats[-1], 2, n_chnls).to(device)
    opt = optim.Adam(chain(embedding_user.parameters(),
                           embedding_item.parameters(),
                           gat.parameters(),
                           rp.parameters()), args.lr)
elif args.model == 'gcmc':
    model = gcmc.GCMC(n_items, n_users, args.in_feats, args.out_feats, \
                      n_chnls, args.reduce_chnl, nonlinear).to(device)
    rp = gcmc.RatingPredictor(args.out_feats, args.n_bases, n_chnls).to(device)
    opt = optim.Adam(chain(model.parameters(), rp.parameters()), args.lr)
else:
    raise RuntimeError()

def partition(uid, iid, rating):
    m = len(uid)
    n_train = int(args.p_train * m)
    n_val = int(args.p_val * m)

    permutation = npr.permutation(m)
    train_idx = permutation[:n_train]
    val_idx = permutation[n_train : n_train + n_val]
    test_idx = permutation[n_train + n_val:]

    train_uid, train_iid, train_rating = uid[train_idx], iid[train_idx], rating[train_idx]
    val_uid, val_iid, val_rating = uid[val_idx], iid[val_idx], rating[val_idx]
    test_uid, test_iid, test_rating = uid[test_idx], iid[test_idx], rating[test_idx]

    return train_uid, train_iid, train_rating, \
           val_uid, val_iid, val_rating, \
           test_uid, test_iid, test_rating

def inv(src, dst):
    _, src_inv, src_deg = np.unique(src, return_counts=True, return_inverse=True)
    _, dst_inv, dst_deg = np.unique(dst, return_counts=True, return_inverse=True)
    brdcstd_src_deg = src_deg[src_inv]
    brdcstd_dst_deg = dst_deg[dst_inv]
    if args.normalization == 'left':
        return 1 / brdcstd_src_deg
    elif args.normalization == 'symmetric':
        return 1 / np.sqrt(brdcstd_src_deg * brdcstd_dst_deg)
    else:
        raise RuntimeError()

def make(uid, iid, dat, dropout_type, dropout_rate):
    if dropout_type is None:
        masks = [dat == i for i in uniq_rating]
        uu = list(map(uid.__getitem__, masks))
        ii = list(map(iid.__getitem__, masks))
        s = [len(x_user), len(x_item)]
        t = [len(x_item), len(x_user)]
        ui = [SubAdj(inv(u, i), u, i, s).to(device) if len(u) > 0 else None for u, i in zip(uu, ii)]
        iu = [SubAdj(inv(i, u), i, u, t).to(device) if len(u) > 0 else None for u, i in zip(uu, ii)]
        return ui, iu
    elif dropout_type == 'node':
        uniq_uid, inv_uid = np.unique(uid, return_inverse=True)
        uniq_iid, inv_iid = np.unique(iid, return_inverse=True)
        uid_mask = (npr.permutation(len(uniq_uid)) > dropout_rate * len(uniq_uid))[inv_uid]
        iid_mask = (npr.permutation(len(uniq_iid)) > dropout_rate * len(uniq_iid))[inv_iid]
        def _make(src, dst, dat, shape):
            masks = [dat == i for i in uniq_rating]
            srcs = map(src.__getitem__, masks)
            dsts = map(dst.__getitem__, masks)
            return [SubAdj(inv(src, dst), src, dst, shape).to(device) \
                    if len(src) > 0 else None for src, dst in zip(srcs, dsts)]
        return _make(uid[iid_mask], iid[iid_mask], dat[iid_mask], [n_users, n_items]), \
               _make(iid[uid_mask], uid[uid_mask], dat[uid_mask], [n_items, n_users])
    elif dropout_type == 'edge':
        raise NotImplementedError()
    else:
        raise RuntimeError()

def sse(r, r_bar):
    arange = th.arange(r_bar.size(1), dtype=th.float, device=device).unsqueeze(0)
    return th.sum((th.sum(F.softmax(r_bar, 1) * arange, 1) - r.float()) ** 2)

def rmse(r, r_bar):
    return (sse(r, r_bar) / len(r)) ** 0.5

writer = SummaryWriter()
# writer = SummaryWriter('runs/' + str(args).replace(' ', ''))

cpu_pttns = list(starmap(partition, zip(sub_uids, sub_iids, sub_ratings)))
gpu_pttns = [list(map(from_numpy, pttn)) for pttn in cpu_pttns]
if args.x:
    x_user = from_numpy(x_user).float()
    x_item = from_numpy(x_item).float()

for i in range(args.n_epochs):
    h_prev = None
    for j, [cpu_pttn, gpu_pttn] in enumerate(zip(cpu_pttns, gpu_pttns)):
        train_uid, train_iid, train_rating, \
        val_uid, val_iid, val_rating, \
        test_uid, test_iid, test_rating = gpu_pttn

        '''
        h_user = embedding_user(th.arange(n_users, device=device))
        h_item = embedding_item(th.arange(n_items, device=device))
        if args.x:
            h_user = th.cat([x_user, h_user], 1)
            h_item = th.cat([x_item, h_item], 1)
        h_user, h_item, _ = gat(u2i, i2u, h_user, h_item, h_prev)
        '''

        u2i, i2u = make(*cpu_pttn[:3], args.dropout_type, args.dropout_rate)
        h_user, h_item, _ = model(u2i, i2u, None, None, h_prev)
        batch_idx = None if args.bs_train is None else \
                    th.randperm(len(train_uid), device=device)[:args.bs]
        train_r = rp(h_user[train_uid[batch_idx]], h_item[train_iid[batch_idx]])
        ce = F.cross_entropy(train_r, train_rating)
        opt.zero_grad()
        ce.backward()
        opt.step()

        u2i, i2u = make(*cpu_pttn[:3], None, 0)
        h_user, h_item, _ = model(u2i, i2u, None, None, h_prev)
        if args.bs_infer is None:
            train_rmse = rmse(train_rating, rp(h_user[train_uid], h_item[train_iid]))
            val_rmse = rmse(val_rating, rp(h_user[val_uid], h_item[val_iid]))
            test_rmse = rmse(test_rating, rp(h_user[test_uid], h_item[test_iid]))
        else:
            def _rmse(rating, uid, iid):
                error = 0
                for i in range(len(rating) // args.bs_infer + 1):
                    m, n = i * args.bs_infer, (i + 1) * args.bs_infer
                    error += sse(rating[m : n], rp(h_user[uid[m : n]], h_item[iid[m : n]]))
                return (error / len(rating)) ** 0.5
            train_rmse = _rmse(train_rating, train_uid, train_iid)
            val_rmse = _rmse(val_rating, val_uid, val_iid)
            test_rmse = _rmse(test_rating, test_uid, test_iid)

        placeholder_i = '0' * (len(str(args.n_epochs)) - len(str(i)))
        placeholder_j = '0' * (len(str(len(sub_uids))) - len(str(j)))
        print('[epoch %s%d][iteration %s%d]ce: %.3e | train rmse: %.3e | val rmse: %.3e | test rmse: %.3e' % (placeholder_i, i, placeholder_j, j, ce.item(), train_rmse.item(), val_rmse.item(), test_rmse.item()))

        global_step = j + i * len(cpu_pttns)
        writer.add_scalar('ce', ce.item(), global_step)
        writer.add_scalar('train rmse', train_rmse.item(), global_step)
        writer.add_scalar('val rmse', val_rmse.item(), global_step)
        writer.add_scalar('test rmse', test_rmse.item(), global_step)

writer.close()
