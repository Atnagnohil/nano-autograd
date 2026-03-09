# 实现网络层构建，目前参考的是Micrograd的nn模型的代码，并且做出了部分优化
import random
import numpy as np
from nanograd.tensor import Tensor
from nanograd.nn.module import Module


class Linear(Module):
    """全连接层: y = x @ weight.T + bias"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.bias_flag = bias
        
        # Kaiming Uniform 初始化（适合 ReLU）
        gain = np.sqrt(2.0)
        bound = gain * np.sqrt(3.0 / in_features)
        
        # weight: (out_features, in_features)
        self.weight = Tensor(
            np.random.uniform(-bound, bound, (out_features, in_features)),
            requires_grad=True
        )
        
        if bias:
            self.bias = Tensor(np.zeros((out_features,), dtype=np.float32), requires_grad=True)
        else:
            self.bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        # x: (batch_size, in_features) 或 (in_features,)
        out = x @ self.weight.T                     # 矩阵乘法
        
        if self.bias is not None:
            out = out + self.bias                   # 自动广播
        return out
    
    def parameters(self):
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params
    
    def __repr__(self):
        return f"Linear(in={self.in_features}, out={self.out_features}, bias={self.bias_flag})"
