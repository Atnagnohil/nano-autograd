# 实现网络层构建，目前参考的是Micrograd的nn模型的代码，并且做出了部分优化
import random
from nanograd.tensor import Tensor

class Module: #基类
    def zero_grad(self): # 梯度清0
        for p in self.parameters():
            p.grad = 0

    def parameters(self): # 返回梯度更新的权重和偏置
        return []
    
    # 让对象可以像函数一样调用
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    
    # 前向传播
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError