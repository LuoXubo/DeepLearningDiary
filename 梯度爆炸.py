import torch
from d2l import torch as d2l
from matplotlib import pyplot as plt

M = torch.normal(0, 1, size=(4,4))
print('一个矩阵 \n', M)

for i in range(100):
    M = torch.mm(M, torch.normal(0, 1, size=(4,4)))

print('乘以100个矩阵之后\n', M)
