import argparse
from itertools import starmap

import numpy as np
from tensorboardX import SummaryWriter
import torch as th
from torch.optim import *
import torch.nn.functional as F

from auto_rec import *
from mf import *
from temporal_mf import *
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--bs-infer', type=int)
parser.add_argument('--bs-train', type=int)
parser.add_argument('--ds', type=str)
parser.add_argument('--gpu', type=int, default='-1')
parser.add_argument('--logdir', type=str)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--n-iters', type=int, required=True)
parser.add_argument('--optim', type=str, required=True)
parser.add_argument('--p-train', type=float, required=True)
parser.add_argument('--p-val', type=float, required=True)
parser.add_argument('--semi', action='store_true')
args = parser.parse_args()

uid = np.load(args.ds + '/uid.npy')
iid = np.load(args.ds + '/iid.npy')
r = np.load(args.ds + '/r.npy')

device = th.device('cpu') if args.gpu < 0 else th.device('cuda:%d' % args.gpu)
perm = th.randperm(len(r), device=device)
uid = th.from_numpy(uid).to(device)[perm]
iid = th.from_numpy(iid).to(device)[perm]
r = th.from_numpy(r).to(device)[perm]
r_max = th.max(r)
r /= r_max
r_mean = th.mean(r)

uids = list(map(th.squeeze, th.split(uid, 1)))
iids = list(map(th.squeeze, th.split(iid, 1)))
rs = list(map(th.squeeze, th.split(r, 1)))
ns_train = [int(args.p_train * len(r)) for r in rs]
ns_val = [int(args.p_val * len(r)) for r in rs]
ns_test = [len(r) - n_train - n_val for r, n_train, n_val in zip(rs, ns_train, ns_val)]
ss = list(zip(ns_train, ns_val, ns_test))
uids_train, uids_val, uids_test = zip(*[th.split(uid, s) for uid, s in zip(uids, ss)])
iids_train, iids_val, iids_test = zip(*[th.split(iid, s) for iid, s in zip(iids, ss)])
rs_train, rs_val, rs_test = zip(*[th.split(r, s) for r, s in zip(rs, ss)])

n_users = th.max(uid) + 1
n_items = th.max(iid) + 1

model = eval(args.model).to(device)
optim = eval(args.optim)

writer = SummaryWriter(args.logdir)

m = None
for i in range(1, len(rs)):
    x = i if args.semi else (i - 1)
    uu, ii, rr = uids[:x] + [uids_train[x]], iids[:x] + [iids_train[x]], rs[:x] + [rs_train[x]]
    vv_train, jj_train, ss_train = uu, ii, rr
    v_val, j_val, s_val = uids_val[x], iids_val[x], rs_val[x]
    v_test, j_test, s_test = uids_test[i], iids_test[i], rs_test[i]
    vv = vv_train[:-1] + [th.cat([vv_train[-1], v_val, v_test])]
    jj = jj_train[:-1] + [th.cat([jj_train[-1], j_val, j_test])]
    ss = ss_train[:-1] + [th.cat([ss_train[-1], s_val, s_test])]

    for j in range(args.n_iters):
        if args.bs_train is None:
            vv_batch, jj_batch, ss_batch = vv_train, jj_train, ss_train
        else:
            vv_batch, jj_batch, ss_batch = [], [], []
            for v_train, j_train, s_train in zip(vv_train, jj_train, ss_train):
                randidx = th.randperm(len(s_train), device=device)[:args.bs_train]
                vv_batch.append(v_train[randidx])
                jj_batch.append(j_train[randidx])
                ss_batch.append(s_train[randidx])

        for p in model.parameters():
            p.requires_grad = True
        tt_batch = model(uu, ii, rr, vv_batch, jj_batch, m)
        mse = F.mse_loss(th.cat(ss_batch[-len(tt_batch):]), th.cat(tt_batch))
        optim.zero_grad()
        mse.backward()
        optim.step()

        for p in model.parameters():
            p.requires_grad = False
        tt = model(uu, ii, rr, vv, jj, m, args.bs_infer)
        t_train = th.cat(([th.cat(tt[:-1])] if len(tt) > 1 else []) + [tt[-1][:len(ss_train[-1])]])
        t_val, t_test = th.split(tt[-1][len(ss_train[-1]):], [len(s_val), len(s_test)])
        rmse_batch = r_max * mse ** 0.5
        rmse_train = r_max * utils.rmse_loss(th.cat(ss_train[-len(tt):]), t_train)
        rmse_val = r_max * utils.rmse_loss(s_val, t_val)
        rmse_test = r_max * utils.rmse_loss(s_test, t_test)

        placeholder_i = '0' * (len(str(len(rs) - 1)) - len(str(i)))
        placeholder_j = '0' * (len(str(args.n_iters)) - len(str(j)))
        print('[%s%d][%s%d]rmse_batch: %.3e | rmse_train: %.3e | rmse_val: %.3e | rmse_test: %.3e' % \
              (placeholder_i, i, placeholder_j, j, rmse_batch, rmse_train, rmse_val, rmse_test))

        global_step = (i - 1) * args.n_iters + j
        writer.add_scalar('rmse_batch', rmse_batch.item(), global_step)
        writer.add_scalar('rmse_train', rmse_train.item(), global_step)
        writer.add_scalar('rmse_val', rmse_val.item(), global_step)
        writer.add_scalar('rmse_test', rmse_test.item(), global_step)

    m = model(uu, ii, rr, None, None, m, detach=True)

writer.close()
