"""测试神经网络层"""
import numpy as np
import pytest
from nanograd.tensor import Tensor
from nanograd.nn import Linear, ReLU, Sigmoid, Tanh


class TestLinear:
    """测试全连接层"""
    
    def test_linear_forward_shape(self):
        """测试前向传播输出形状"""
        layer = Linear(10, 5)
        x = Tensor(np.random.randn(3, 10))
        out = layer(x)
        
        assert out.shape == (3, 5), f"期望形状 (3, 5)，得到 {out.shape}"
    
    def test_linear_forward_no_bias(self):
        """测试无偏置的全连接层"""
        layer = Linear(10, 5, bias=False)
        x = Tensor(np.random.randn(3, 10))
        out = layer(x)
        
        assert out.shape == (3, 5)
        assert layer.bias is None
    
    def test_linear_parameters(self):
        """测试参数管理"""
        layer = Linear(10, 5, bias=True)
        params = layer.parameters()
        
        assert len(params) == 2, "应该有 weight 和 bias 两个参数"
        assert params[0].shape == (5, 10), "weight 形状应该是 (out, in)"
        assert params[1].shape == (5,), "bias 形状应该是 (out,)"
    
    def test_linear_backward(self):
        """测试反向传播梯度计算"""
        layer = Linear(3, 2, bias=True)
        x = Tensor(np.array([[1.0, 2.0, 3.0]]), requires_grad=True)
        
        # 前向传播
        out = layer(x)
        
        # 反向传播
        loss = out.sum()
        loss.backward()
        
        # 检查梯度是否计算
        assert x.grad is not None, "输入应该有梯度"
        assert layer.weight.grad is not None, "权重应该有梯度"
        assert layer.bias.grad is not None, "偏置应该有梯度"
        
        # 检查梯度形状
        assert x.grad.shape == x.shape
        assert layer.weight.grad.shape == layer.weight.shape
        assert layer.bias.grad.shape == layer.bias.shape
    
    def test_linear_gradient_values(self):
        """测试梯度数值正确性（数值梯度检验）"""
        layer = Linear(2, 1, bias=False)
        
        # 手动设置权重为已知值
        layer.weight.data = np.array([[1.0, 2.0]])
        
        x = Tensor(np.array([[3.0, 4.0]]), requires_grad=True)
        
        # 前向传播: out = x @ weight.T = [3, 4] @ [[1], [2]] = [11]
        out = layer(x)
        
        # 反向传播
        out.backward()
        
        # 检查输入梯度: dL/dx = weight = [1, 2]
        expected_x_grad = np.array([[1.0, 2.0]])
        np.testing.assert_allclose(x.grad, expected_x_grad, rtol=1e-5)
        
        # 检查权重梯度: dL/dw = x.T = [[3], [4]]
        expected_w_grad = np.array([[3.0, 4.0]])
        np.testing.assert_allclose(layer.weight.grad, expected_w_grad, rtol=1e-5)


class TestActivations:
    """测试激活函数"""
    
    def test_relu_forward(self):
        """测试 ReLU 前向传播"""
        relu = ReLU()
        x = Tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]))
        out = relu(x)
        
        expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
        np.testing.assert_allclose(out.data, expected, rtol=1e-5)
    
    def test_relu_backward(self):
        """测试 ReLU 反向传播"""
        relu = ReLU()
        x = Tensor(np.array([-2.0, -1.0, 0.0, 1.0, 2.0]), requires_grad=True)
        out = relu(x)
        
        loss = out.sum()
        loss.backward()
        
        # ReLU 导数：x > 0 时为 1，否则为 0
        expected_grad = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
        np.testing.assert_allclose(x.grad, expected_grad, rtol=1e-5)
    
    def test_sigmoid_forward(self):
        """测试 Sigmoid 前向传播"""
        sigmoid = Sigmoid()
        x = Tensor(np.array([0.0]))
        out = sigmoid(x)
        
        # sigmoid(0) = 0.5
        np.testing.assert_allclose(out.data, 0.5, rtol=1e-5)
    
    def test_sigmoid_backward(self):
        """测试 Sigmoid 反向传播"""
        sigmoid = Sigmoid()
        x = Tensor(np.array([0.0]), requires_grad=True)
        out = sigmoid(x)
        
        out.backward()
        
        # sigmoid'(0) = sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5 = 0.25
        np.testing.assert_allclose(x.grad, 0.25, rtol=1e-5)
    
    def test_tanh_forward(self):
        """测试 Tanh 前向传播"""
        tanh = Tanh()
        x = Tensor(np.array([0.0]))
        out = tanh(x)
        
        # tanh(0) = 0
        np.testing.assert_allclose(out.data, 0.0, rtol=1e-5)
    
    def test_tanh_backward(self):
        """测试 Tanh 反向传播"""
        tanh = Tanh()
        x = Tensor(np.array([0.0]), requires_grad=True)
        out = tanh(x)
        
        out.backward()
        
        # tanh'(0) = 1 - tanh^2(0) = 1 - 0 = 1
        np.testing.assert_allclose(x.grad, 1.0, rtol=1e-5)


class TestMultiLayerNetwork:
    """测试多层网络组合"""
    
    def test_simple_mlp(self):
        """测试简单的多层感知机"""
        # 构建网络: Linear(2, 3) -> ReLU -> Linear(3, 1)
        layer1 = Linear(2, 3)
        relu = ReLU()
        layer2 = Linear(3, 1)
        
        # 前向传播
        x = Tensor(np.random.randn(5, 2), requires_grad=True)
        h = layer1(x)
        h = relu(h)
        out = layer2(h)
        
        assert out.shape == (5, 1), f"期望输出形状 (5, 1)，得到 {out.shape}"
        
        # 反向传播
        loss = out.sum()
        loss.backward()
        
        # 检查所有参数都有梯度
        assert layer1.weight.grad is not None
        assert layer1.bias.grad is not None
        assert layer2.weight.grad is not None
        assert layer2.bias.grad is not None
        assert x.grad is not None
    
    def test_zero_grad(self):
        """测试梯度清零"""
        layer = Linear(2, 1)
        x = Tensor(np.random.randn(1, 2))
        
        # 第一次前向和反向传播
        out = layer(x)
        out.sum().backward()
        
        # 保存梯度
        grad_before = layer.weight.grad.copy()
        
        # 清零梯度
        layer.zero_grad()
        
        # 检查梯度是否清零
        assert np.allclose(layer.weight.grad, 0.0)
        assert np.allclose(layer.bias.grad, 0.0)
    
    def test_parameters_collection(self):
        """测试参数收集"""
        layer1 = Linear(10, 5)
        layer2 = Linear(5, 2, bias=False)
        
        params1 = layer1.parameters()
        params2 = layer2.parameters()
        
        assert len(params1) == 2  # weight + bias
        assert len(params2) == 1  # weight only
        
        # 所有参数都应该是 Tensor 且 requires_grad=True
        for p in params1 + params2:
            assert isinstance(p, Tensor)
            assert p.requires_grad


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
