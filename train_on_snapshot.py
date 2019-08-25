import argparse

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
parser.add_argument('--x-train', type=str, requried=True)
parser.add_argument('--x-val', type=str, requried=True)
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

uids, iids, rs = th.split(uid, 0), th.split(iid, 0), th.split(r, 0)
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

for i in range(1, len(rs)):
    for j in range(args.n_iters):
        x_train = lambda x: eval(args.x_train.replace('x', x))
        uu_train, ii_train, rr_train = map(x_train, ['uu', 'ii', 'rr'])
        x_val = lambda x: eval(arrgs.x_val.replace('x', x))
        u_val, i_val, r_val = map(th.cat, map(x_val, ['uu', 'ii', 'rr']))
        u_test, i_test, r_test = map(th.cat, [uids_test[i], iids_test[i], rs_test[i]])

        if args.bs_train is None:
            u_batch, i_batch, r_batch = u_train, i_train, r_train
        else:
            perm_batch = th.randperm(len(r_train), device=device)[:args.bs_train]
            u_batch, i_batch, r_batch = u_train[perm_batch], i_train[perm_batch], r_train[perm_batch]

        for p in model.parameters():
            p.requires_grad = True
        s_batch = model(u_train, i_train, r_train, u_batch, i_batch)
        mse = F.mse_loss(r_batch, s_batch)
        optim.zero_grad()
        mse.backward()
        optim.step()

        for p in model.parameters():
            p.requires_grad = False
        u_cat = th.cat([u_train, u_val, u_test])
        i_cat = th.cat([i_train, i_val, i_test])
        s = model(u_train, i_train, r_train, u_cat, i_cat, args.bs_infer)
        s_train, s_val, s_test = th.split(s, [len(r_train), len(r_val), len(r_test)])
        rmse_batch = r_max * mse ** 0.5
        rmse_train = r_max * utils.rmse_loss(r_train, s_train)
        rmse_val = r_max * utils.rmse_loss(r_val, s_val)
        rmse_test = r_max * utils.rmse_loss(r_test, s_test)

        '''
        placeholder = '0' * (len(str(args.n_iters)) - len(str(i + 1)))
        print('[%s%d]rmse_batch: %.3e | rmse_train: %.3e | rmse_val: %.3e | rmse_test: %.3e' % \
              (placeholder, i + 1, rmse_batch, rmse_train, rmse_val, rmse_test))

        writer.add_scalar('rmse_batch', rmse_batch.item(), i + 1)
        writer.add_scalar('rmse_train', rmse_train.item(), i + 1)
        writer.add_scalar('rmse_val', rmse_val.item(), i + 1)
        writer.add_scalar('rmse_test', rmse_test.item(), i + 1)
        '''

writer.close()
