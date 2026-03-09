"""测试 Tensor 类的张量运算和梯度计算"""
import numpy as np
import torch
from nanograd.tensor import Tensor


def test_reshape():
    """测试 reshape 操作"""
    # Nanograd
    a = Tensor([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    b = a.reshape(3, 2)
    c = b.sum()
    c.backward()
    
    # PyTorch
    a_torch = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, requires_grad=True)
    b_torch = a_torch.reshape(3, 2)
    c_torch = b_torch.sum()
    c_torch.backward()
    
    assert np.allclose(b.data, b_torch.detach().numpy())
    assert np.allclose(a.grad, a_torch.grad.numpy())


def test_transpose():
    """测试转置操作"""
    # Nanograd
    a = Tensor([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    b = a.transpose()  # (3, 2)
    c = b.sum()
    c.backward()
    
    # PyTorch
    a_torch = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, requires_grad=True)
    b_torch = a_torch.T
    c_torch = b_torch.sum()
    c_torch.backward()
    
    assert np.allclose(b.data, b_torch.detach().numpy())
    assert np.allclose(a.grad, a_torch.grad.numpy())


def test_sum_with_axis():
    """测试带 axis 的 sum"""
    # Nanograd
    a = Tensor([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    b = a.sum(axis=0)  # (3,)
    c = b.sum()
    c.backward()
    
    # PyTorch
    a_torch = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, requires_grad=True)
    b_torch = a_torch.sum(dim=0)
    c_torch = b_torch.sum()
    c_torch.backward()
    
    assert np.allclose(b.data, b_torch.detach().numpy())
    assert np.allclose(a.grad, a_torch.grad.numpy())


def test_mean():
    """测试 mean 操作"""
    # Nanograd
    a = Tensor([[1, 2, 3], [4, 5, 6]])
    b = a.mean()
    b.backward()
    
    # PyTorch
    a_torch = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, requires_grad=True)
    b_torch = a_torch.mean()
    b_torch.backward()
    
    assert np.allclose(b.data, b_torch.item())
    assert np.allclose(a.grad, a_torch.grad.numpy())


def test_mean_with_axis():
    """测试带 axis 的 mean"""
    # Nanograd
    a = Tensor([[1, 2, 3], [4, 5, 6]])
    b = a.mean(axis=1)  # (2,)
    c = b.sum()
    c.backward()
    
    # PyTorch
    a_torch = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, requires_grad=True)
    b_torch = a_torch.mean(dim=1)
    c_torch = b_torch.sum()
    c_torch.backward()
    
    assert np.allclose(b.data, b_torch.detach().numpy())
    assert np.allclose(a.grad, a_torch.grad.numpy())


def test_division():
    """测试除法"""
    # Nanograd
    a = Tensor([6.0, 8.0, 10.0])
    b = Tensor([2.0, 4.0, 5.0])
    c = a / b
    d = c.sum()
    d.backward()
    
    # PyTorch
    a_torch = torch.tensor([6.0, 8.0, 10.0], requires_grad=True)
    b_torch = torch.tensor([2.0, 4.0, 5.0], requires_grad=True)
    c_torch = a_torch / b_torch
    d_torch = c_torch.sum()
    d_torch.backward()
    
    assert np.allclose(c.data, c_torch.detach().numpy())
    assert np.allclose(a.grad, a_torch.grad.numpy())
    assert np.allclose(b.grad, b_torch.grad.numpy())


def test_sigmoid():
    """测试 Sigmoid 激活函数"""
    # Nanograd
    a = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    b = a.sigmoid()
    c = b.sum()
    c.backward()
    
    # PyTorch
    a_torch = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    b_torch = torch.sigmoid(a_torch)
    c_torch = b_torch.sum()
    c_torch.backward()
    
    assert np.allclose(b.data, b_torch.detach().numpy(), atol=1e-6)
    assert np.allclose(a.grad, a_torch.grad.numpy(), atol=1e-6)


def test_tanh():
    """测试 Tanh 激活函数"""
    # Nanograd
    a = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    b = a.tanh()
    c = b.sum()
    c.backward()
    
    # PyTorch
    a_torch = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)
    b_torch = torch.tanh(a_torch)
    c_torch = b_torch.sum()
    c_torch.backward()
    
    assert np.allclose(b.data, b_torch.detach().numpy(), atol=1e-6)
    assert np.allclose(a.grad, a_torch.grad.numpy(), atol=1e-6)


def test_complex_chain():
    """测试复杂的链式运算"""
    # Nanograd
    x = Tensor([[1, 2], [3, 4]])
    W = Tensor([[0.5, 0.3], [0.2, 0.7]])
    b = Tensor([0.1, -0.2])
    
    # y = (x @ W + b).relu().mean()
    y = (x @ W + b).relu().mean()
    y.backward()
    
    # PyTorch
    x_torch = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, requires_grad=True)
    W_torch = torch.tensor([[0.5, 0.3], [0.2, 0.7]], dtype=torch.float32, requires_grad=True)
    b_torch = torch.tensor([0.1, -0.2], dtype=torch.float32, requires_grad=True)
    
    y_torch = torch.relu(x_torch @ W_torch + b_torch).mean()
    y_torch.backward()
    
    assert np.allclose(y.data, y_torch.item(), atol=1e-6)
    assert np.allclose(x.grad, x_torch.grad.numpy(), atol=1e-6)
    assert np.allclose(W.grad, W_torch.grad.numpy(), atol=1e-6)
    assert np.allclose(b.grad, b_torch.grad.numpy(), atol=1e-6)


def test_batch_matmul():
    """测试批量矩阵乘法"""
    # Nanograd
    x = Tensor([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    W = Tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])  # (3, 2)
    y = x @ W  # (2, 2)
    loss = y.sum()
    loss.backward()
    
    # PyTorch
    x_torch = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, requires_grad=True)
    W_torch = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=torch.float32, requires_grad=True)
    y_torch = x_torch @ W_torch
    loss_torch = y_torch.sum()
    loss_torch.backward()
    
    assert np.allclose(y.data, y_torch.detach().numpy())
    assert np.allclose(x.grad, x_torch.grad.numpy())
    assert np.allclose(W.grad, W_torch.grad.numpy())


def test_subtraction():
    """测试减法"""
    # Nanograd
    a = Tensor([5, 10, 15])
    b = Tensor([1, 2, 3])
    c = a - b
    d = c.sum()
    d.backward()
    
    # PyTorch
    a_torch = torch.tensor([5, 10, 15], dtype=torch.float32, requires_grad=True)
    b_torch = torch.tensor([1, 2, 3], dtype=torch.float32, requires_grad=True)
    c_torch = a_torch - b_torch
    d_torch = c_torch.sum()
    d_torch.backward()
    
    assert np.allclose(c.data, c_torch.detach().numpy())
    assert np.allclose(a.grad, a_torch.grad.numpy())
    assert np.allclose(b.grad, b_torch.grad.numpy())


def test_zero_grad():
    """测试梯度清零"""
    a = Tensor([1, 2, 3])
    b = a * 2
    c = b.sum()
    c.backward()
    
    assert not np.allclose(a.grad, 0)
    
    a.zero_grad()
    assert np.allclose(a.grad, 0)


def test_reshape_with_minus_one():
    """测试 reshape 中的 -1"""
    # Nanograd
    a = Tensor([[1, 2, 3], [4, 5, 6]])  # (2, 3)
    b = a.reshape(-1, 2)  # (3, 2)
    c = b.sum()
    c.backward()
    
    # PyTorch
    a_torch = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32, requires_grad=True)
    b_torch = a_torch.reshape(-1, 2)
    c_torch = b_torch.sum()
    c_torch.backward()
    
    assert np.allclose(b.data, b_torch.detach().numpy())
    assert np.allclose(a.grad, a_torch.grad.numpy())
