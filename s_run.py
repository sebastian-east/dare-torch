import torch
import numpy as np
from riccati import dare
from test_riccati import initialize

def solve(A, B, Q, R):
    n = A.shape[0]
    
    G = B @ torch.inverse(R) @ B.T
    AinvT = torch.inverse(A).T
    Z1 = A + G @ AinvT @ Q
    Z2 = - G @ AinvT
    Z3 = - AinvT @ Q
    Z4 = AinvT
    Z12 = torch.cat((Z1, Z2), dim = 1)
    Z34 = torch.cat((Z3, Z4), dim = 1)
    Z = torch.cat((Z12, Z34), dim = 0)
    val, vec = torch.eig(Z, eigenvectors = True)
    val = torch.norm(val, dim=1)
    val, ind = torch.sort(val)
    vec = vec[:, ind]
            
    return vec[n:,:n] * torch.inverse(vec[:n,:n])

A, B, Q, R = initialize()
DARE = dare()
P = DARE(A, B, Q, R)
print(P)
A = A.clone().double()
P = solve(A, B, Q, R)
print(P)