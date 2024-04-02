import torch
import numpy as np
import math
from scipy.stats import ortho_group
import torch.nn.functional as F


### generate random vectors for input data from gaussian distribution
def generate_random_vec(num, dim, mean = None, std = None):
    if mean is None:
        mean = torch.zeros(dim)
    if std is None:
        std = torch.ones(dim)
    return (torch.randn(num, dim) * std + mean).float()

# generate random vectors follow https://arxiv.org/abs/1711.05174 
def generate_random_vec_zeyuan(num, dim):
    # generate num*dim orthonormal matrix U
    p = dim//2
    n2 = num // 2
    
    U1 = ortho_group.rvs(n2)[:, :p]
    V1 = ortho_group.rvs(p)
    S1 = np.diag(np.array([1.0 / (i + 1) for i in range(p)]))
    X1 = U1 @ S1 @ V1
    
    U2 = ortho_group.rvs(n2)[:, :p]
    V2 = ortho_group.rvs(p)
    S2 = np.diag(np.array([2.0 / (i + 10) for i in range(p)]))
    X2 = U2 @ S2 @ V2
    
    ### X = [X1 0; 0 X2] block matrix
    X = np.zeros((num, dim))
    X[:n2, :p] = X1
    X[n2:, p:] = X2
    
    return torch.tensor(X)

### generate sparse random matrix
def sparse_random_matrix(dim1, dim2, sparsity_ratio = 0.01):
    with torch.no_grad():
        A = torch.randn(dim1, dim2)
        A = F.dropout(A, p = 1 - sparsity_ratio)
        
    return A


# modified from https://arxiv.org/abs/1711.05174, randn can generate almost orthogonal matrix when dim is large
def generate_random_vec_fast_zeyuan(num, dim):
    # generate num*dim orthonormal matrix U
    p = dim//2
    n2 = num // 2
    
    U1 = torch.randn(n2, p)
    V1 = torch.randn(p, p)
    S1 = torch.diag(torch.tensor([1.0 / (i + 1) for i in range(p)]))
    X1 = U1 @ S1 @ V1
    
    U2 = torch.randn(n2, p)
    V2 = torch.randn(p, p)
    S2 = torch.diag(torch.tensor([1.0 / (i + 1) for i in range(p)]))
    X2 = U2 @ S2 @ V2
    
    ### X = [X1 0; 0 X2] block matrix
    X = np.zeros((num, dim))
    X[:n2, :p] = X1
    X[n2:, p:] = X2
    
    return torch.tensor(X)
