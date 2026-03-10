"""张量运算操作"""
import numpy as np


def _ensure_tensor(x):
    """确保输入是 Tensor 类型"""
    from .tensor import Tensor
    return x if isinstance(x, Tensor) else Tensor(x)


# ==================== 基础运算 ====================

def add(a, b):
    """加法: a + b"""
    from .tensor import Tensor
    
    a = _ensure_tensor(a)
    b = _ensure_tensor(b)
    out = Tensor(a.data + b.data, (a, b), '+')
    
    def _backward():
        if a.requires_grad:
            a.grad += Tensor._unbroadcast(out.grad, a.shape)
        if b.requires_grad:
            b.grad += Tensor._unbroadcast(out.grad, b.shape)
    
    out._backward = _backward
    return out


def sub(a, b):
    """减法: a - b"""
    from .tensor import Tensor
    
    a = _ensure_tensor(a)
    b = _ensure_tensor(b)
    out = Tensor(a.data - b.data, (a, b), '-')
    
    def _backward():
        if a.requires_grad:
            a.grad += Tensor._unbroadcast(out.grad, a.shape)
        if b.requires_grad:
            b.grad += Tensor._unbroadcast(-out.grad, b.shape)
    
    out._backward = _backward
    return out


def mul(a, b):
    """乘法: a * b(逐元素)"""
    from .tensor import Tensor
    
    a = _ensure_tensor(a)
    b = _ensure_tensor(b)
    out = Tensor(a.data * b.data, (a, b), '*')
    
    def _backward():
        if a.requires_grad:
            grad_a = b.data * out.grad
            a.grad += Tensor._unbroadcast(grad_a, a.shape)
        if b.requires_grad:
            grad_b = a.data * out.grad
            b.grad += Tensor._unbroadcast(grad_b, b.shape)
    
    out._backward = _backward
    return out


def div(a, b):
    """除法: a / b"""
    from .tensor import Tensor
    
    a = _ensure_tensor(a)
    b = _ensure_tensor(b)
    out = Tensor(a.data / b.data, (a, b), '/')
    
    def _backward():
        if a.requires_grad:
            grad_a = out.grad / b.data
            a.grad += Tensor._unbroadcast(grad_a, a.shape)
        if b.requires_grad:
            grad_b = -out.grad * a.data / (b.data ** 2)
            b.grad += Tensor._unbroadcast(grad_b, b.shape)
    
    out._backward = _backward
    return out


def pow(a, exponent):
    """幂运算: a ** exponent"""
    from .tensor import Tensor
    
    assert isinstance(exponent, (int, float)), "只支持标量指数"
    a = _ensure_tensor(a)
    out = Tensor(a.data ** exponent, (a,), f'**{exponent}')
    
    def _backward():
        if a.requires_grad:
            a.grad += exponent * (a.data ** (exponent - 1)) * out.grad
    
    out._backward = _backward
    return out


def neg(a):
    """取负: -a"""
    from .tensor import Tensor
    
    a = _ensure_tensor(a)
    out = Tensor(-a.data, (a,), 'neg')
    
    def _backward():
        if a.requires_grad:
            a.grad += -out.grad
    
    out._backward = _backward
    return out


# ==================== 矩阵运算 ====================

def matmul(a, b):
    """矩阵乘法: a @ b"""
    from .tensor import Tensor
    
    a = _ensure_tensor(a)
    b = _ensure_tensor(b)
    out = Tensor(a.data @ b.data, (a, b), '@')
    
    def _backward():
        if a.requires_grad:
            grad_a = out.grad @ b.data.T
            a.grad += Tensor._unbroadcast(grad_a, a.shape)
        if b.requires_grad:
            grad_b = a.data.T @ out.grad
            b.grad += Tensor._unbroadcast(grad_b, b.shape)
    
    out._backward = _backward
    return out


# ==================== 聚合运算 ====================

def sum(a, axis=None, keepdims=False):
    """求和"""
    from .tensor import Tensor
    
    a = _ensure_tensor(a)
    out = Tensor(a.data.sum(axis=axis, keepdims=keepdims), (a,), 'sum')
    
    def _backward():
        if a.requires_grad:
            grad = out.grad
            # 如果 sum 降维了，需要恢复形状
            if axis is not None:
                if not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                grad = np.broadcast_to(grad, a.shape)
            else:
                grad = np.broadcast_to(grad, a.shape)
            a.grad += grad
    
    out._backward = _backward
    return out


def mean(a, axis=None, keepdims=False):
    """求平均值"""
    from .tensor import Tensor
    
    a = _ensure_tensor(a)
    out = Tensor(a.data.mean(axis=axis, keepdims=keepdims), (a,), 'mean')
    
    def _backward():
        if a.requires_grad:
            grad = out.grad
            # 计算平均值的元素数量
            if axis is None:
                n = a.data.size
            else:
                n = a.data.shape[axis] if isinstance(axis, int) else np.prod([a.data.shape[i] for i in axis])
            
            # 恢复形状
            if axis is not None:
                if not keepdims:
                    grad = np.expand_dims(grad, axis=axis)
                grad = np.broadcast_to(grad, a.shape)
            else:
                grad = np.broadcast_to(grad, a.shape)
            
            a.grad += grad / n
    
    out._backward = _backward
    return out


# ==================== 形状操作 ====================

def reshape(a, shape):
    """改变形状"""
    from .tensor import Tensor
    
    a = _ensure_tensor(a)
    # 处理 -1 的情况
    if isinstance(shape, tuple) and -1 in shape:
        shape = list(shape)
        idx = shape.index(-1)
        shape[idx] = a.data.size // (-np.prod(shape))
        shape = tuple(shape)
    
    out = Tensor(a.data.reshape(shape), (a,), 'reshape')
    
    def _backward():
        if a.requires_grad:
            a.grad += out.grad.reshape(a.shape)
    
    out._backward = _backward
    return out


def transpose(a, axes=None):
    """转置"""
    from .tensor import Tensor
    
    a = _ensure_tensor(a)
    out = Tensor(a.data.transpose(axes), (a,), 'transpose')
    
    def _backward():
        if a.requires_grad:
            # 反向转置
            if axes is None:
                a.grad += out.grad.T
            else:
                # 计算逆转置
                inv_axes = np.argsort(axes)
                a.grad += out.grad.transpose(inv_axes)
    
    out._backward = _backward
    return out


# ==================== 激活函数 ====================

def relu(a):
    """ReLU 激活函数"""
    from .tensor import Tensor
    
    a = _ensure_tensor(a)
    out = Tensor(np.maximum(0, a.data), (a,), 'relu')
    
    def _backward():
        if a.requires_grad:
            a.grad += (a.data > 0) * out.grad
    
    out._backward = _backward
    return out


def sigmoid(a):
    """Sigmoid 激活函数"""
    from .tensor import Tensor
    
    a = _ensure_tensor(a)
    sig = 1 / (1 + np.exp(-a.data))
    out = Tensor(sig, (a,), 'sigmoid')
    
    def _backward():
        if a.requires_grad:
            # sigmoid 的导数：sigmoid(x) * (1 - sigmoid(x))
            a.grad += out.data * (1 - out.data) * out.grad
    
    out._backward = _backward
    return out


def tanh(a):
    """Tanh 激活函数"""
    from .tensor import Tensor
    
    a = _ensure_tensor(a)
    out = Tensor(np.tanh(a.data), (a,), 'tanh')
    
    def _backward():
        if a.requires_grad:
            # tanh 的导数：1 - tanh^2(x)
            a.grad += (1 - out.data ** 2) * out.grad
    
    out._backward = _backward
    return out


# ==================== 其他张量组合操作 ====================

def stack(tensors, axis=0):
    """沿着新轴堆叠多个 Tensor, 类似 np.stack
    
    Args:
        tensors: Tensor 的列表或元组
        axis: 插入新维度的位置（支持负数索引）
    
    Returns:
        堆叠后的新 Tensor
    """
    from .tensor import Tensor
    
    # 输入检查与转换
    if not isinstance(tensors, (list, tuple)):
        raise TypeError("tensors 必须是 list 或 tuple")
    
    if len(tensors) == 0:
        raise ValueError("tensors 列表不能为空")
    
    # 全部转为 Tensor，并记录原始 requires_grad 状态
    ts = [_ensure_tensor(t) for t in tensors]
    
    # 检查所有 Tensor 形状一致（np.stack 会强制要求）
    first_shape = ts[0].shape
    for t in ts[1:]:
        if t.shape != first_shape:
            raise ValueError(f"所有 Tensor 必须有相同形状，得到 {t.shape} != {first_shape}")
    
    # 前向计算
    datas = [t.data for t in ts]
    out_data = np.stack(datas, axis=axis)
    
    # 是否需要梯度（只要有一个需要，就都需要传播）
    requires_grad = any(t.requires_grad for t in ts)
    
    # 创建输出 Tensor
    out = Tensor(
        out_data,
        _children=tuple(ts),
        _op=f"stack_axis{axis}",
        requires_grad=requires_grad
    )
    
    def _backward():
        if not out.requires_grad:
            return
        
        # 把输出梯度沿着 stack 的轴拆分
        # np.split 会自动处理 axis 的负索引
        grad_splits = np.split(out.grad, len(ts), axis=axis)
        
        for t, grad_part in zip(ts, grad_splits):
            if t.requires_grad:
                # grad_part 保留了被 split 的维度（大小为 1），需要 squeeze 掉
                grad_part = np.squeeze(grad_part, axis=axis)
                
                if t.grad is None:
                    t.grad = np.zeros_like(t.data)
                t.grad += grad_part   # 支持梯度累加
    
    out._backward = _backward
    return out