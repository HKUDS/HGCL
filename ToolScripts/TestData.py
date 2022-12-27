import numpy as np 
import scipy.sparse as sp

import torch.utils.data as data

import pickle

class TstData(data.Dataset):
    def __init__(self, coomat, trnMat):
        super(TstData, self).__init__()
        self.csrmat = (trnMat.tocsr() != 0) * 1.0
        tstLocs = [None] * coomat.shape[0]
        tstUsrs = set()
        for i in range(len(coomat.data)):#将所有用户的测试集交互商品的索引存在列表里
            row = coomat.row[i]
            col = coomat.col[i]
            if tstLocs[row] is None:
                tstLocs[row] = list()
            tstLocs[row].append(col)
            tstUsrs.add(row)
        tstUsrs = np.array(list(tstUsrs))
        self.tstUsrs = tstUsrs
        self.tstLocs = tstLocs#所有用户的测试正例的位置

    def __len__(self):
        return len(self.tstUsrs)
    def __getitem__(self, idx):
        return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])