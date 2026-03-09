"""测试 Tensor 类的自动微分功能（标量和张量），并与 PyTorch 对比验证"""
import numpy as np
import torch
from nanograd.engine import Tensor


def test_scalar_add():
    """测试标量加法"""
    # Nanograd
    a = Tensor(2.0)
    b = Tensor(3.0)
    c = a + b
    c.backward()
    
    # PyTorch
    a_torch = torch.tensor(2.0, requires_grad=True)
    b_torch = torch.tensor(3.0, requires_grad=True)
    c_torch = a_torch + b_torch
    c_torch.backward()
    
    assert np.allclose(c.data, c_torch.item())
    assert np.allclose(a.grad, a_torch.grad.item())
    assert np.allclose(b.grad, b_torch.grad.item())


def test_scalar_mul():
    """测试标量乘法"""
    # Nanograd
    a = Tensor(2.0)
    b = Tensor(3.0)
    c = a * b
    c.backward()
    
    # PyTorch
    a_torch = torch.tensor(2.0, requires_grad=True)
    b_torch = torch.tensor(3.0, requires_grad=True)
    c_torch = a_torch * b_torch
    c_torch.backward()
    
    assert np.allclose(c.data, c_torch.item())
    assert np.allclose(a.grad, a_torch.grad.item())
    assert np.allclose(b.grad, b_torch.grad.item())


def test_scalar_pow():
    """测试标量幂运算"""
    # Nanograd
    a = Tensor(3.0)
    b = a ** 2
    b.backward()
    
    # PyTorch
    a_torch = torch.tensor(3.0, requires_grad=True)
    b_torch = a_torch ** 2
    b_torch.backward()
    
    assert np.allclose(b.data, b_torch.item())
    assert np.allclose(a.grad, a_torch.grad.item())


def test_scalar_complex():
    """测试复杂标量表达式: (a + b) * (a - b)"""
    # Nanograd
    a = Tensor(5.0)
    b = Tensor(3.0)
    c = (a + b) * (a - b)
    c.backward()
    
    # PyTorch
    a_torch = torch.tensor(5.0, requires_grad=True)
    b_torch = torch.tensor(3.0, requires_grad=True)
    c_torch = (a_torch + b_torch) * (a_torch - b_torch)
    c_torch.backward()
    
    assert np.allclose(c.data, c_torch.item())
    assert np.allclose(a.grad, a_torch.grad.item())
    assert np.allclose(b.grad, b_torch.grad.item())


def test_vector_add():
    """测试向量加法"""
    # Nanograd
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    c = a + b
    c_sum = c.sum()
    c_sum.backward()
    
    # PyTorch
    a_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    b_torch = torch.tensor([4.0, 5.0, 6.0], requires_grad=True)
    c_torch = a_torch + b_torch
    c_sum_torch = c_torch.sum()
    c_sum_torch.backward()
    
    assert np.allclose(c.data, c_torch.detach().numpy())
    assert np.allclose(a.grad, a_torch.grad.numpy())
    assert np.allclose(b.grad, b_torch.grad.numpy())


def test_vector_mul():
    """测试向量逐元素乘法"""
    # Nanograd
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([2.0, 3.0, 4.0])
    c = a * b
    c_sum = c.sum()
    c_sum.backward()
    
    # PyTorch
    a_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    b_torch = torch.tensor([2.0, 3.0, 4.0], requires_grad=True)
    c_torch = a_torch * b_torch
    c_sum_torch = c_torch.sum()
    c_sum_torch.backward()
    
    assert np.allclose(c.data, c_torch.detach().numpy())
    assert np.allclose(a.grad, a_torch.grad.numpy())
    assert np.allclose(b.grad, b_torch.grad.numpy())


def test_matrix_matmul():
    """测试矩阵乘法"""
    # Nanograd
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])
    c = a @ b
    c_sum = c.sum()
    c_sum.backward()
    
    # PyTorch
    a_torch = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b_torch = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    c_torch = a_torch @ b_torch
    c_sum_torch = c_torch.sum()
    c_sum_torch.backward()
    
    assert np.allclose(c.data, c_torch.detach().numpy())
    assert np.allclose(a.grad, a_torch.grad.numpy())
    assert np.allclose(b.grad, b_torch.grad.numpy())


def test_relu():
    """测试 ReLU 激活函数"""
    # Nanograd
    a = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    b = a.relu()
    b_sum = b.sum()
    b_sum.backward()
    
    # PyTorch
    a_torch = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    b_torch = torch.relu(a_torch)
    b_sum_torch = b_torch.sum()
    b_sum_torch.backward()
    
    assert np.allclose(b.data, b_torch.detach().numpy())
    assert np.allclose(a.grad, a_torch.grad.numpy())


def test_broadcast_add():
    """测试广播加法"""
    # Nanograd
    a = Tensor([[1.0, 2.0, 3.0]])  # shape (1, 3)
    b = Tensor([[1.0], [2.0], [3.0]])  # shape (3, 1)
    c = a + b  # 广播到 (3, 3)
    c_sum = c.sum()
    c_sum.backward()
    
    # PyTorch
    a_torch = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    b_torch = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
    c_torch = a_torch + b_torch
    c_sum_torch = c_torch.sum()
    c_sum_torch.backward()
    
    assert np.allclose(c.data, c_torch.detach().numpy())
    assert np.allclose(a.grad, a_torch.grad.numpy())
    assert np.allclose(b.grad, b_torch.grad.numpy())


def test_broadcast_mul():
    """测试广播乘法"""
    # Nanograd
    a = Tensor([1.0, 2.0, 3.0])  # shape (3,)
    b = Tensor(2.0)  # 标量
    c = a * b
    c_sum = c.sum()
    c_sum.backward()
    
    # PyTorch
    a_torch = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    b_torch = torch.tensor(2.0, requires_grad=True)
    c_torch = a_torch * b_torch
    c_sum_torch = c_torch.sum()
    c_sum_torch.backward()
    
    assert np.allclose(c.data, c_torch.detach().numpy())
    assert np.allclose(a.grad, a_torch.grad.numpy())
    assert np.allclose(b.grad, b_torch.grad.numpy())


def test_complex_expression():
    """测试复杂表达式: y = (x @ W + b).relu().sum()"""
    # Nanograd
    x = Tensor([[1.0, 2.0]])  # (1, 2)
    W = Tensor([[0.5, 0.3], [0.2, 0.7]])  # (2, 2)
    b = Tensor([0.1, -0.5])  # (2,)
    
    y = (x @ W + b).relu()
    loss = y.sum()
    loss.backward()
    
    # PyTorch
    x_torch = torch.tensor([[1.0, 2.0]], requires_grad=True)
    W_torch = torch.tensor([[0.5, 0.3], [0.2, 0.7]], requires_grad=True)
    b_torch = torch.tensor([0.1, -0.5], requires_grad=True)
    
    y_torch = torch.relu(x_torch @ W_torch + b_torch)
    loss_torch = y_torch.sum()
    loss_torch.backward()
    
    assert np.allclose(y.data, y_torch.detach().numpy(), atol=1e-5)
    assert np.allclose(x.grad, x_torch.grad.numpy(), atol=1e-5)
    assert np.allclose(W.grad, W_torch.grad.numpy(), atol=1e-5)
    assert np.allclose(b.grad, b_torch.grad.numpy(), atol=1e-5)
