# 实现随机梯度下降
import numpy as np 
from nanograd.optim.optimizer import Optimizer
from nanograd.tensor import Tensor


class SGD(Optimizer):
    """
    随机梯度下降 SGD 优化器
    支持学习率 + 可选动量 (momentum)
    数学公式：
    v = momentum * v + grad
    param.data -= lr * v
    """
    def __init__(self, params: list[Tensor], lr: float = 1e-3, momentum: float = 0.0):
        super().__init__(params)
        self.lr = lr  # 学习率
        self.momentum = momentum
        # 维护一个速度缓存区 [velocity buffer]
        self.velocity_buffer = [
            np.zeros_like(p.data) if self.momentum > 0 else None
            for p in self.params
        ]
    
    def step(self):
        """更新"""
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad

            if self.momentum > 0:
                self.velocity_buffer[i] = self.momentum * self.velocity_buffer[i] + grad
                grad = self.velocity_buffer[i]  # 将梯度更新为基类后的梯度

            p.data -= self.lr * grad


