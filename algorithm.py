import numpy as np
import torch
from munkres import Munkres
from tqdm import tqdm
from scipy.sparse import csr_matrix

def greedy_hungarian(D,device):
    P = torch.zeros_like(D)
    size = D.shape[0]
    index_S = [i for i in range(size)]
    index_S_hat = [i for i in range(size)]
    for i in range(size):
        cur_size = D.shape[0]
        argmin = torch.argmin(D.to(device)).item()
        r = argmin // cur_size
        c = argmin % cur_size
        P[index_S[r]][index_S_hat[c]] = 1
        index_S.remove(index_S[r])
        index_S_hat.remove(index_S_hat[c])
        D = D[torch.arange(D.size(0)) != r]
        D = D.t()[torch.arange(D.t().size(0)) != c].t()
    return P.t()

def hungarian(D):
    P = torch.zeros_like(D)
    matrix = D.tolist()
    m = Munkres()
    indexes = m.compute(matrix)
    total = 0
    for r,c in indexes:
        P[r][c] = 1
        total += matrix[r][c]
    return P.t()

def approximate_NN(S_emb, S_hat_emb):
    index_S = torch.squeeze(torch.argsort(S_emb, dim=0)).cpu()
    index_S_hat = torch.squeeze(torch.argsort(S_hat_emb, dim=0)).cpu()
    n = index_S.shape[0]
    P = torch.zeros(n,n)
    for i in range(P.shape[0]):
        P[index_S_hat[i]][index_S[i]] = 1
    return P





