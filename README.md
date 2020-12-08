# differentiable-dare

Implementation of a differetiable discrete-time algebraic Riccati equation (DARE) solver in PyTorch.

Solves DARE equations of the form

![equation](https://latex.codecogs.com/svg.latex?P%20%3D%20A%5E%5Ctop%20P%20A%20-%20A%5E%5Ctop%20P%20B%20%28%20R%20&plus;%20B%5E%5Ctop%20P%20B%20%29%5E%7B-1%7D%20B%5E%5Ctop%20P%20A%20&plus;%20Q)

for the matrix P, and computes the derivatives of P with respect to input matrices A, B, Q, and R.

Details on the method for differentiating the DARE are available in this paper: [Infinite-Horizon Differentiable Model Predictive Control](https://arxiv.org/pdf/2001.02244.pdf)


## Usage
```
from riccati import dare
```
Forwards solution (currently uses the scipy DARE solver and is not PyTorch native).
```
DARE = dare()
P = DARE(A, B, C, D)
```
Derivative (implemented in native PyTorch code).
```
P = P.sum()
P.backward()
```
Requires PyTorch, numpy, and scipy.
