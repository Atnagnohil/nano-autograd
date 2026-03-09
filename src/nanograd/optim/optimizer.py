# 实现优化器基类，参考的是tinygrad项目
import numpy as np
from typing import List


class Optimizer:
    def __init__(self, params: List['Tensor']):
        self.params = [p for p in params if getattr(p, 'requires_grad', False)]
        assert len(self.params) > 0, "优化器必须至少有一个requires_grad=True的参数"

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.fill(0.0)

    def step(self):
        raise NotImplementedError("子类必须实现step()方法")


