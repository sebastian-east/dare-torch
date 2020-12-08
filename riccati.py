import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import solve_discrete_are

def V_pert(m,n):
    """ Form the V_{m,n} perturbation matrix as defined in the paper

    Args:
        m
        n

    Returns:
        V_{m,n}
    """
    V = torch.zeros((m*n,m*n))
    for i in range(m*n):
        block = ((i*m) - ((i*m) % (m*n))) / (m*n)
        col = (i*m) % (m*n)
        V[i,col + round(block)] = 1
    return V

def vec(A):
    """returns vec(A) of matrix A (i.e. columns stacked into a vector)

    Args:
        A

    Returns:
        vec(A)
    """
    m, n = A.shape
    vecA = torch.zeros((m*n, 1))
    for i in range(n):
        vecA[i*m:(i+1)*m,:] = A[:,i].unsqueeze(1)

    return vecA #torch.reshape(A, (m * n, 1)) #A.view(m*n, 1) # vecA

def inv_vec(v,A):
    """Inverse operation of vecA"""
    v_out = torch.zeros_like(A)
    m, n = A.shape
    for i in range(n):
        v_out[:,i] = v[0,i*m:(i+1)*m]
        
    return v_out #torch.reshape(v, (m, n)).T #v.view(m, n).T #v_out

def kronecker(A, B):
    """Kronecker product of matrices A and B"""
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))

class Riccati(torch.autograd.Function):

    @staticmethod                       #FORWARDS PASS
    def forward(ctx, A, B, Q, R):
        
        if not (A.type() == B.type() and A.type() == Q.type() and A.type() == R.type()):
            raise Exception('A, B, Q, and R must be of the same type.')
            
        Atemp = A.detach().numpy()
        Btemp = B.detach().numpy()
        Q = 0.5 * (Q + Q.transpose(0, 1))
        Qtemp = Q.detach().numpy()
        R = 0.5 * (R + R.transpose(0,1))
        Rtemp = R.detach().numpy()

        P = solve_discrete_are(Atemp, Btemp, Qtemp, Rtemp)
        P = torch.from_numpy(P).type(A.type())

        ctx.save_for_backward(P, A, B, Q, R) #Save variables for
        #backwards pass
        return P

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = vec(grad_output).transpose(0,1).double()
        P, A, B, Q, R = ctx.saved_tensors
        n, m = B.shape
        #Computes derivatives using method detailed in paper
        
        M3 = R + B.transpose(0,1) @ P @ B
        M2 = M3.inverse()
        M1 = P - P @ B @ M2 @ B.transpose(0,1) @ P

        LHS = kronecker(B.transpose(0,1), B.transpose(0,1))
        LHS = kronecker(M2, M2) @ LHS
        LHS = kronecker(P @ B, P @ B) @ LHS
        LHS = LHS - kronecker(torch.eye(n), P@B@M2@B.transpose(0,1))
        LHS = LHS - kronecker(P @ B @ M2 @ B.transpose(0,1), torch.eye(n))
        LHS = LHS + torch.eye(n ** 2)
        LHS = kronecker(A.transpose(0,1), A.transpose(0,1)) @ LHS
        LHS = torch.eye(n ** 2) - LHS
        invLHS = torch.inverse(LHS)

        RHS = V_pert(n,n).type(A.type()) + torch.eye(n ** 2)
        RHS = RHS @ kronecker(torch.eye(n), A.transpose(0,1) @ M1)
        dA = invLHS @ RHS
        dA = grad_output @ dA
        dA = inv_vec(dA, A)

        RHS = kronecker(torch.eye(m), B.transpose(0,1) @ P)
        RHS = (torch.eye(m ** 2) + V_pert(m,m).type(A.type())) @ RHS
        RHS = -kronecker(M2, M2) @ RHS
        RHS = -kronecker(P@B, P@B) @ RHS
        RHS = RHS - (torch.eye(n ** 2) + V_pert(n,n).type(A.type())) @ (kronecker(P @ B @ M2, P))
        RHS = kronecker(A.transpose(0,1), A.transpose(0,1)) @ RHS
        dB = invLHS @ RHS
        dB = grad_output @ dB
        dB = inv_vec(dB, B)

        RHS = torch.eye(n ** 2).double()
        dQ = invLHS @ RHS
        dQ = grad_output @ dQ
        dQ = inv_vec(dQ, Q)
        dQ = 0.5 * (dQ + dQ.transpose(0, 1))

        RHS = -kronecker(M2, M2)
        RHS = - kronecker(P @ B, P @ B) @ RHS
        RHS = kronecker(A.transpose(0,1), A.transpose(0,1)) @ RHS
        dR = invLHS @ RHS
        dR = grad_output @ dR
        dR = inv_vec(dR, R)
        dR = 0.5 * (dR + dR.transpose(0, 1))

        return dA, dB, dQ, dR

class dare(nn.Module):

    def __init__(self):
        super(dare, self).__init__()

    def forward(self, A, B, Q, R):
        return Riccati.apply(A, B, Q, R)
