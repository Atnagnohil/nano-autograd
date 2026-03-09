"""Tensor 类：支持自动微分的多维数组"""
import numpy as np


class Tensor:
    """支持自动微分的张量类
    
    基于 NumPy 实现，支持：
    - 多维数组运算
    - 自动梯度计算
    - 广播机制
    """
    
    def __init__(self, data, _children=(), _op="", requires_grad=True):
        """初始化 Tensor
        
        Args:
            data: 数据，可以是标量、列表或 NumPy 数组
            _children: 父节点（用于构建计算图）
            _op: 操作符名称
            requires_grad: 是否需要计算梯度
        """
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._parents = tuple(_children)
        self._op = _op
    
    @property
    def shape(self):
        """返回张量形状"""
        return self.data.shape
    
    @property
    def ndim(self):
        """返回张量维度数"""
        return self.data.ndim
    
    def zero_grad(self):
        """清零梯度"""
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)
    
    def backward(self):
        """反向传播计算梯度"""
        if not self.requires_grad:
            raise RuntimeError("Tensor does not require grad")
        
        # 拓扑排序
        topo = []
        visited = set()
        
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for parent in node._parents:
                    build_topo(parent)
                topo.append(node)
        
        build_topo(self)
        
        # 初始化输出梯度为 1
        self.grad = np.ones_like(self.data)
        
        # 反向传播
        for node in reversed(topo):
            node._backward()
    
    @staticmethod
    def _unbroadcast(grad, original_shape):
        """将广播后的梯度还原到原始形状
        
        Args:
            grad: 广播后的梯度
            original_shape: 原始形状
            
        Returns:
            还原后的梯度
        """
        # 处理维度增加的情况
        ndims_added = grad.ndim - len(original_shape)
        for _ in range(ndims_added):
            grad = grad.sum(axis=0)
        
        # 处理维度为 1 被广播的情况
        for i, dim in enumerate(original_shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        
        return grad.reshape(original_shape)
    
    # ==================== 运算符重载 ====================
    
    def __add__(self, other):
        from .ops import add
        return add(self, other)
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        from .ops import sub
        return sub(self, other)
    
    def __rsub__(self, other):
        from .ops import sub
        other = other if isinstance(other, Tensor) else Tensor(other)
        return sub(other, self)
    
    def __mul__(self, other):
        from .ops import mul
        return mul(self, other)
    
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        from .ops import div
        return div(self, other)
    
    def __rtruediv__(self, other):
        from .ops import div
        other = other if isinstance(other, Tensor) else Tensor(other)
        return div(other, self)
    
    def __pow__(self, other):
        from .ops import pow
        return pow(self, other)
    
    def __neg__(self):
        from .ops import neg
        return neg(self)
    
    def __matmul__(self, other):
        from .ops import matmul
        return matmul(self, other)
    
    # ==================== 张量操作 ====================
    
    def sum(self, axis=None, keepdims=False):
        """求和"""
        from .ops import sum
        return sum(self, axis, keepdims)
    
    def mean(self, axis=None, keepdims=False):
        """求平均值"""
        from .ops import mean
        return mean(self, axis, keepdims)
    
    def reshape(self, *shape):
        """改变形状"""
        from .ops import reshape
        return reshape(self, shape)
    
    def transpose(self, *axes):
        """转置"""
        from .ops import transpose
        return transpose(self, axes if axes else None)
    
    @property
    def T(self):
        """转置（属性）"""
        return self.transpose()
    
    @staticmethod
    def stack(tensors, axis=0):
        """沿着指定轴堆叠多个 Tensor"""
        from .ops import stack
        return stack(tensors, axis=axis)

    # ==================== 激活函数 ====================
    
    def relu(self):
        """ReLU 激活函数"""
        from .ops import relu
        return relu(self)
    
    def sigmoid(self):
        """Sigmoid 激活函数"""
        from .ops import sigmoid
        return sigmoid(self)
    
    def tanh(self):
        """Tanh 激活函数"""
        from .ops import tanh
        return tanh(self)
    
    # ==================== 其他 ====================
    
    def __repr__(self):
        return f"Tensor(shape={self.shape}, op='{self._op}')"
    
    def __str__(self):
        return f"Tensor({self.data})"
