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
