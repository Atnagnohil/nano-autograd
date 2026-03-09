"""Engine 模块：导出 Tensor 类供向后兼容"""
from .tensor import Tensor

# 为了向后兼容，也导出为 Value
Value = Tensor

__all__ = ['Tensor', 'Value']
