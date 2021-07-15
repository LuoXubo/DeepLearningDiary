import torch
from torch import nn
from torch.nn import functional as F

# 生成一个网络，包含一个具有256个单元和ReLU激活函数的全连接的隐藏层，然后是一个具有10个隐藏单元且不带激活函数的全连接的输出层
# nn.Sequential定义了一种特殊的Module，即在PyTorch中表示一个块的类
# net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

X = torch.rand(2, 20)
# 自定义块
class MLP(nn.Module):
    # 用模型参数声明层。这里声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类 Block 的构造函数来执行必要的初始化
        super().__init__()
        self.hidden = nn.Linear(20, 256) # 隐藏层
        self.out = nn.Linear(256, 10) # 输出层

    # 定义模型的正向传播，即如何根据输入'x'返回所需要的模型输出
    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))

# net = MLP()
# print(net(X))

# 顺序块
# Sequential的设计是为了把其他模块串起来。
# 为了构建我们自己的简化的MySequential，我们只需要定义两个关键函数：
# 1. 一种将块逐个追加到列表中的函数。
# 2. 一种正向传播函数，用于将输入按追加块的顺序传递给块组成的“链条”。
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for block in args:
            self._modules[block] = block
    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X
# net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
# print(net(X))

# 在正向传播函数中执行代码
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 不计算梯度的随即权重参数，因此其在训练期间保持不变
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)
    def forward(self, X):
        X = self.linear(X)
        # 使用创建的常量参数以及relu和dot函数
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        # 复用全连接层，相当于两个全连接层共享参数
        X = self.linear(X)
        # 控制流
        while(X.abs().sum() > 1):
            X /= 2
        return X.sum

# net = FixedHiddenMLP()
# print(net(X))

class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
print(chimera(X))
