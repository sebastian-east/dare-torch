from riccati import Riccati, dare
import torch
import numpy as np
from scipy.linalg import expm

def initialize():
    # define continuous time SS sytem matrices
    A_c = -np.eye(3)
    B_c = np.zeros((3, 2))
    B_c[[0,1,0], [0,1,0]] = 1
    
    # convert SS system to discrete time over sampling perior 'dt'
    dt = 0.1
    A = expm(A_c*dt)
    B = (A - np.identity(3)) @ np.linalg.inv(A_c) @ B_c
    
    # set cost at single timestep
    Q = np.eye(3)
    R = np.eye(2) 
    R[0,0] = 0.1
        
    A = torch.tensor(A, requires_grad=True)
    B = torch.tensor(B, requires_grad=True)
    Q = torch.from_numpy(Q)
    R = torch.from_numpy(R)
    
    A = A.clone().detach().double().requires_grad_(True)
    B = B.clone().detach().double().requires_grad_(True)
    Q = Q.clone().detach().double().requires_grad_(True)
    R = R.clone().detach().double().requires_grad_(True)
    
    return A, B, Q, R


def test_gradients():

    A, B, Q, R = initialize()
    
    Riccati.apply(A.float(), B.float(), Q.float(), R.float())
    
    riccati = Riccati.apply
    
    from torch.autograd import gradcheck
    input = (A, B, Q, R)
    test = gradcheck(riccati, input, eps=1E-6, atol=1E-4, raise_exception=True)
    
def test_interface():
    
    A, B, Q, R = initialize()
    
    P = Riccati.apply(A, B, Q, R)
    P = P.sum()
    P.backward()
    
    DARE = dare()
    P = DARE(A, B, Q, R)
    P = P.sum()
    P.backward()
