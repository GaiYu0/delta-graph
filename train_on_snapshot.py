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
    k = i if args.semi else (i - 1)
    uu, ii, rr = uids[:k] + [uids_train[k]], iids[:k] + [iids_train[k]], rs[:k] + [rs[k]]
    vv_train, jj_train, ss_train = uu, ii, rr
    if args.bs_train is None:
        vv_batch, jj_batch, ss_batch = vv, jj, ss
    else:
        vv_batch, jj_batch, ss_batch = [], [], []
        for _u, _i, _r in zip(uu, ii, rr):
            randidx = th.randperm(len(r), device=device)[:args.bs_train]
            vv_batch.append(_u[randidx])
            jj_batch.append(_i[randidx])
            ss_batch.append(_r[randidx])
    vv_val, jj_val, ss_val = [uids_val[k]], [iids_val[k]], [rs_val[k]]
    vv_test, jj_test, ss_test = [uids_test[i]], [iids_test[i]], [rs_test[i]]
    vv = list(map(th.cat, zip(vv_train, vv_val, vv_test)))
    jj = list(map(th.cat, zip(jj_train, jj_val, jj_test)))
    ss = list(map(th.cat, zip(ss_train, ss_val, ss_test)))

    for j in range(args.n_iters):
        for p in model.parameters():
            p.requires_grad = True
        tt_batch = model(uu, ii, rr, vv_batch, jj_batch, m)
        mse = F.mse_loss(th.cat(ss_batch), th.cat(tt_batch))
        optim.zero_grad()
        mse.backward()
        optim.step()

        for p in model.parameters():
            p.requires_grad = False

        tt = model(uu, ii, rr, vv, jj, m, args.bs_infer)
        tt_train, tt_val, tt_batch = zip(*[th.split(t, [len(s_train), len(s_val), len(s_test)]) \
                                           for t, s_train, s_val, s_test in zip(tt,
                                                                                ss_train,
                                                                                ss_val,
                                                                                ss_test)])
        rmse_batch = r_max * mse ** 0.5
        rmse_train = r_max * utils.rmse_loss(th.cat(ss_train), th.cat(tt_train))
        rmse_val = r_max * utils.rmse_loss(th.cat(ss_val), th.cat(tt_val))
        rmse_test = r_max * utils.rmse_loss(th.cat(ss_test), th.cat(tt_test))

        placeholder = '0' * (len(str(args.n_iters)) - len(str(i + 1)))
        print('[%s%d]rmse_batch: %.3e | rmse_train: %.3e | rmse_val: %.3e | rmse_test: %.3e' % \
              (placeholder, i + 1, rmse_batch, rmse_train, rmse_val, rmse_test))

        '''
        writer.add_scalar('rmse_batch', rmse_batch.item(), i + 1)
        writer.add_scalar('rmse_train', rmse_train.item(), i + 1)
        writer.add_scalar('rmse_val', rmse_val.item(), i + 1)
        writer.add_scalar('rmse_test', rmse_test.item(), i + 1)
        '''
    m = model(uu, ii, rr, None, None, m, detach=True)

writer.close()