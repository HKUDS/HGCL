import torch as t
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import numpy as np
# import scipy.sparse as sp

def showSparseTensor(tensor):
    index = t.nonzero(tensor)
    countArr = t.sum(tensor!=0, dim=1).cpu().numpy()
    start=0
    end=0
    tmp = tensor[index[:,0], index[:,1]].cpu().detach().numpy()
    for i in countArr:
        start = end
        end += i
        print(tmp[start: end])

def masked_softmax(vec, mask, dim=1, epsilon=1e-6):
    exps = t.exp(vec)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
    ret = (masked_exps/masked_sums)
    return ret

def list2Str(s):
    ret = str(s[0])
    for i in range(1, len(s)):
        ret = ret + '_' + str(s[i])
    return ret

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    if type(sparse_mx) != sp.coo_matrix:
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


class sampleLargeGraph(nn.Module):
    def __init__(self):
        super(sampleLargeGraph, self).__init__()
        self.flag = 0
    

    def makeMask(self, nodes, size):
        mask = np.ones(size)
        if not nodes is None:
            mask[nodes] = 0.0
        return mask
        
    def updateBdgt(self, adj, nodes):
        if nodes is None:
            return 0
        # tembat = 1000
        # ret = 0
        # for i in range(int(np.ceil(len(nodes) / tembat))):
        #     st = tembat * i
        #     ed = min((i+1) * tembat, len(nodes))
        #     temNodes = nodes[st: ed]
        ret = np.sum(adj[nodes], axis=0).A #消掉0维
        return ret
        
    def sample(self, budget, mask, sampNum):
        score = (mask * np.reshape(np.array(budget), [-1])) ** 2
        norm = np.sum(score)
        if norm == 0:
            return np.random.choice(len(score), 1)
        score = list(score / norm)
        arrScore = np.array(score)
        posNum = np.sum(np.array(score)!=0)
        if posNum < sampNum:
            pckNodes1 = np.squeeze(np.argwhere(arrScore!=0))
            pckNodes2 = np.random.choice(np.squeeze(np.argwhere(arrScore==0.0)), min(len(score) - posNum, sampNum - posNum), replace=False)
            pckNodes = np.concatenate([pckNodes1, pckNodes2], axis=0)
        else:
            pckNodes = np.random.choice(len(score), sampNum, p=score, replace=False)
        return pckNodes

    def forward(self, graph_adj, pickNodes, sampDepth=2, sampNum=20000, Flage=False):
        nodeMask = self.makeMask(pickNodes, graph_adj.shape[0])
        nodeBdgt = self.updateBdgt(graph_adj, pickNodes) #用来做垂直的normalize

        for i in range(sampDepth):
            newNodes = self.sample(nodeBdgt, nodeMask, sampNum)
            nodeMask = nodeMask * self.makeMask(newNodes, graph_adj.shape[0])
            if i == sampDepth - 1:
                break
            nodeBdgt += self.updateBdgt(graph_adj, newNodes)
        pckNodes = np.reshape(np.argwhere(nodeMask==0), [-1])
        return pckNodes


class sampleHeterLargeGraph(nn.Module):
    def __init__(self):
        super(sampleHeterLargeGraph, self).__init__()
        self.flag = 1
    

    def makeMask(self, nodes, size):
        mask = np.ones(size)
        if not nodes is None:
            mask[nodes] = 0.0
        return mask
        
    def updateBdgt(self, adj, nodes):
        if nodes is None:
            return 0
        # tembat = 1000
        # ret = 0
        # for i in range(int(np.ceil(len(nodes) / tembat))):
        #     st = tembat * i
        #     ed = min((i+1) * tembat, len(nodes))
        #     temNodes = nodes[st: ed]
        ret = np.sum(adj[nodes], axis=0).A #消掉0维
        return ret
        
    def sample(self, budget, mask, sampNum):
        score = (mask * np.array(budget).reshape(-1)) ** 2
        norm = np.sum(score)
        if norm == 0:
            return np.random.choice(len(score), 1)
        score = list(score / norm)
        arrScore = np.array(score)
        posNum = np.sum(np.array(score)!=0)
        if posNum < sampNum:
            pckNodes1 = np.squeeze(np.argwhere(arrScore!=0))
            pckNodes2 = np.random.choice(np.squeeze(np.argwhere(arrScore==0.0)), min(len(score) - posNum, sampNum - posNum), replace=False)
            pckNodes = np.concatenate([pckNodes1, pckNodes2], axis=0)
        else:
            pckNodes = np.random.choice(len(score), sampNum, p=score, replace=False)
        return pckNodes

    def forward(self, graph_adj, pickNodes, pickNodes2=None, sampDepth=2, sampNum=10000):
        #pickNodes  user dimension
        #pickNodes2 item dimension
        if self.flag:
            print('sample depth:', sampDepth, '     sample Num:', sampNum)
            self.flag = 0
        nodeMask1 = self.makeMask(pickNodes, graph_adj.shape[0]) 
        nodeMask2 = self.makeMask(pickNodes2, graph_adj.shape[1])
        nodeBdgt2 = self.updateBdgt(graph_adj, pickNodes) 
        if pickNodes2 is None:
            pickNodes2 = self.sample(nodeBdgt2, nodeMask2, len(pickNodes))
            nodeMask2 = nodeMask2 * self.makeMask(pickNodes2, graph_adj.shape[1])
        nodeBdgt1 = self.updateBdgt(graph_adj.T, pickNodes2)
        for i in range(sampDepth):
            newNodes1 = self.sample(nodeBdgt1, nodeMask1, sampNum)
            nodeMask1 = nodeMask1 * self.makeMask(newNodes1, graph_adj.shape[0])
            newNodes2 = self.sample(nodeBdgt2, nodeMask2, sampNum)
            nodeMask2 = nodeMask2 * self.makeMask(newNodes2, graph_adj.shape[1])
            if i == sampDepth - 1:
                break
            nodeBdgt2 += self.updateBdgt(graph_adj, newNodes1)
            nodeBdgt1 += self.updateBdgt(graph_adj.T, newNodes2)
        pckNodes1 = np.reshape(np.where(nodeMask1==0), [-1])
        pckNodes2 = np.reshape(np.where(nodeMask2==0), [-1])
        return pckNodes1, pckNodes2

# import pickle
# if __name__ == "__main__":
#     with open(r'dataset/CiaoDVD/data.pkl', 'rb') as fs:
#         data = pickle.load(fs) 
#     trainMat, _, _, _, _ = data
#     sample_graph_nodes = sampleHeterLargeGraph()
#     seed_nodes = np.random.choice(trainMat.shape[0], 1000)
#     sample_graph_nodes(trainMat, seed_nodes)
