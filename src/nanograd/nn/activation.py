# 添加激活函数
from nanograd.nn.module import Module
from nanograd.tensor import Tensor

class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()
    def parameters(self):
        return []
    def __repr__(self):
        return "ReLU()"

class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()
    def parameters(self):
        return []
    def __repr__(self):
        return "Sigmoid()"

class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()
    def parameters(self):
        return []
    def __repr__(self):
        return "Tanh()"