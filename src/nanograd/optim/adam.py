# 实现 Adam 优化器
import numpy as np 
from nanograd.optim.optimizer import Optimizer
from nanograd.tensor import Tensor


class Adam(Optimizer):
    """Adam 优化器
    
    算法：
    m = beta1 * m + (1-beta1) * grad
    v = beta2 * v + (1-beta2) * grad²
    m_hat = m / (1 - beta1^t)  # 偏差修正
    v_hat = v / (1 - beta2^t)  # 偏差修正
    param.data -= lr * m_hat / (sqrt(v_hat) + eps)
    """
    def __init__(self, params: list[Tensor], lr: float = 1e-3, betas: tuple = (0.9, 0.999), eps: float = 1e-8):
        super().__init__(params)
        self.lr = lr  # 学习率
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0  # timestep

    def step(self):
        """更新参数"""
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            grad = p.grad
            
            # 更新一阶矩估计和二阶矩估计
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad * grad
            
            # 偏差修正
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # 更新参数
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


