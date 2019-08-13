import numpy as np
import scipy.sparse as sps
import torch as th
import torch.sparse as ths

class Adj:
    def __init__(self, data, i, j, shape):
        self.scipy_coo = sps.coo_matrix((data, (i, j)), shape=shape)
        self.row = th.from_numpy(self.scipy_coo.row).long()
        self.col = th.from_numpy(self.scipy_coo.col).long()

        self.scipy_csr = self.scipy_coo.tocsr()
        self.indices = th.from_numpy(self.scipy_csr.indices).long()
        self.indptr = th.from_numpy(self.scipy_csr.indptr).long()

        dat = th.from_numpy(data).float()
        idx = th.from_numpy(np.vstack([i, j])).long()
        self.coo = ths.FloatTensor(idx, dat, shape).coalesce()

    def to(self, device):
        self.row = self.row.to(device)
        self.col = self.col.to(device)
        self.indices = self.indices.to(device)
        self.indptr = self.indptr.to(device)
        self.coo = self.coo.to(device)
        return self

class SubAdj(Adj):
    def __init__(self, data, i, j, shape):
        self.shape = shape
        uniq_i, inv_i = np.unique(i, return_inverse=True)
        uniq_j, inv_j = np.unique(j, return_inverse=True)
        row = np.arange(len(uniq_i))[inv_i]
        col = np.arange(len(uniq_j))[inv_j]
        self.uniq_i = th.from_numpy(uniq_i)
        self.uniq_j = th.from_numpy(uniq_j)
        super().__init__(data, row, col, [len(uniq_i), len(uniq_j)])

    def to(self, device):
        super().to(device)
        self.uniq_i = self.uniq_i.to(device)
        self.uniq_j = self.uniq_j.to(device)
        return self
