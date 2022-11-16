# dare-torch

Implementation of a differetiable discrete-time algebraic Riccati equation (DARE) solver in PyTorch.

Solves DARE equations of the form

![equation](https://latex.codecogs.com/svg.latex?P%20%3D%20A%5E%5Ctop%20P%20A%20-%20A%5E%5Ctop%20P%20B%20%28%20R%20&plus;%20B%5E%5Ctop%20P%20B%20%29%5E%7B-1%7D%20B%5E%5Ctop%20P%20A%20&plus;%20Q)

for the matrix P, and computes the derivatives of P with respect to input matrices A, B, Q, and R.

Details on the method for differentiating the DARE are available in this paper: [Infinite-Horizon Differentiable Model Predictive Control](https://arxiv.org/pdf/2001.02244.pdf)

## License

This code is available under the unlicense: do whatever you want with it. If you find it useful in your research, I would appreciate it if you could include the following citation in any resulting publications:

```
@article{East2020,
  title={Infinite-Horizon Differentiable Model Predictive Control},
  author={Sebastian East and Marco Gallieri and Jonathan Masci and Jan Koutn{\'i}k and Mark Cannon},
  journal={ArXiv},
  year={2020},
  volume={abs/2001.02244}
}
```

## Numerical tests for validity of derivative

![Numerical Tests](https://github.com/sebastian-east/dare-torch/workflows/Numerical%20Tests/badge.svg?branch=main) <- this failure is caused by a version conflict in the required packages that has arisen since I wrote the code; the derivative is still working correctly when run locally. I'll fix the github action when I've got some time available.

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
