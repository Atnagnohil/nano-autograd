"""测试优化器模块"""
import numpy as np
import pytest
from nanograd.tensor import Tensor
from nanograd.optim import SGD, Adam


class TestSGD:
    """测试 SGD 优化器"""
    
    def test_sgd_basic(self):
        """测试基本的 SGD（无动量）"""
        # 创建一个简单的参数
        x = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        
        # 手动设置梯度
        x.grad = np.array([0.1, 0.2, 0.3])
        
        # 创建优化器
        optimizer = SGD([x], lr=0.1)
        
        # 保存初始值
        initial_data = x.data.copy()
        
        # 执行一步优化
        optimizer.step()
        
        # 验证更新：x_new = x_old - lr * grad
        expected = initial_data - 0.1 * np.array([0.1, 0.2, 0.3])
        np.testing.assert_allclose(x.data, expected, rtol=1e-5)
    
    def test_sgd_with_momentum(self):
        """测试带动量的 SGD"""
        x = Tensor([1.0, 2.0], requires_grad=True)
        optimizer = SGD([x], lr=0.1, momentum=0.9)
        
        # 第一步
        x.grad = np.array([1.0, 1.0])
        initial_data = x.data.copy()
        optimizer.step()
        
        # v = 0.9 * 0 + 1.0 = 1.0
        # x = x - 0.1 * 1.0 = x - 0.1
        expected_step1 = initial_data - 0.1 * 1.0
        np.testing.assert_allclose(x.data, expected_step1, rtol=1e-5)
        
        # 第二步
        x.grad = np.array([1.0, 1.0])
        data_after_step1 = x.data.copy()
        optimizer.step()
        
        # v = 0.9 * 1.0 + 1.0 = 1.9
        # x = x - 0.1 * 1.9 = x - 0.19
        expected_step2 = data_after_step1 - 0.1 * 1.9
        np.testing.assert_allclose(x.data, expected_step2, rtol=1e-5)
    
    def test_sgd_zero_grad(self):
        """测试梯度清零"""
        x = Tensor([1.0, 2.0], requires_grad=True)
        x.grad = np.array([0.5, 0.5])
        
        optimizer = SGD([x], lr=0.1)
        optimizer.zero_grad()
        
        # 验证梯度被清零
        np.testing.assert_array_equal(x.grad, np.zeros_like(x.data))
    
    def test_sgd_multiple_params(self):
        """测试多个参数的优化"""
        x1 = Tensor([1.0], requires_grad=True)
        x2 = Tensor([2.0], requires_grad=True)
        
        x1.grad = np.array([0.1])
        x2.grad = np.array([0.2])
        
        optimizer = SGD([x1, x2], lr=0.5)
        
        initial_x1 = x1.data.copy()
        initial_x2 = x2.data.copy()
        
        optimizer.step()
        
        np.testing.assert_allclose(x1.data, initial_x1 - 0.5 * 0.1, rtol=1e-5)
        np.testing.assert_allclose(x2.data, initial_x2 - 0.5 * 0.2, rtol=1e-5)


class TestAdam:
    """测试 Adam 优化器"""
    
    def test_adam_basic(self):
        """测试基本的 Adam 优化"""
        x = Tensor([1.0, 2.0], requires_grad=True)
        x.grad = np.array([0.1, 0.2])
        
        optimizer = Adam([x], lr=0.01, betas=(0.9, 0.999), eps=1e-8)
        
        initial_data = x.data.copy()
        optimizer.step()
        
        # 第一步：
        # m = 0.9 * 0 + 0.1 * grad = 0.1 * grad
        # v = 0.999 * 0 + 0.001 * grad^2 = 0.001 * grad^2
        # m_hat = m / (1 - 0.9^1) = m / 0.1 = grad
        # v_hat = v / (1 - 0.999^1) = v / 0.001 = grad^2
        # update = lr * m_hat / (sqrt(v_hat) + eps)
        
        m = 0.1 * x.grad
        v = 0.001 * x.grad ** 2
        m_hat = m / (1 - 0.9)
        v_hat = v / (1 - 0.999)
        expected = initial_data - 0.01 * m_hat / (np.sqrt(v_hat) + 1e-8)
        
        np.testing.assert_allclose(x.data, expected, rtol=1e-5)
    
    def test_adam_multiple_steps(self):
        """测试 Adam 多步优化"""
        x = Tensor([1.0], requires_grad=True)
        optimizer = Adam([x], lr=0.1, betas=(0.9, 0.999))
        
        # 执行多步
        for _ in range(3):
            x.grad = np.array([1.0])
            optimizer.step()
        
        # 验证参数确实在更新
        assert x.data[0] < 1.0, "参数应该减小"
    
    def test_adam_zero_grad(self):
        """测试梯度清零"""
        x = Tensor([1.0, 2.0], requires_grad=True)
        x.grad = np.array([0.5, 0.5])
        
        optimizer = Adam([x], lr=0.01)
        optimizer.zero_grad()
        
        np.testing.assert_array_equal(x.grad, np.zeros_like(x.data))
    
    def test_adam_bias_correction(self):
        """测试偏差修正的效果"""
        x = Tensor([1.0], requires_grad=True)
        x.grad = np.array([1.0])
        
        optimizer = Adam([x], lr=0.1, betas=(0.9, 0.999))
        
        # 第一步
        initial = x.data.copy()
        optimizer.step()
        
        # 验证偏差修正确实在工作
        # t=1 时，m_hat = m / (1 - 0.9^1) = m / 0.1
        # 这意味着第一步的更新会被放大
        expected_m = (1 - 0.9) * 1.0  # 0.1
        expected_v = (1 - 0.999) * 1.0  # 0.001
        expected_m_hat = expected_m / (1 - 0.9)  # 1.0
        expected_v_hat = expected_v / (1 - 0.999)  # 1.0
        expected_update = 0.1 * expected_m_hat / (np.sqrt(expected_v_hat) + 1e-8)
        expected_value = initial - expected_update
        
        np.testing.assert_allclose(x.data, expected_value, rtol=1e-5)


class TestOptimizerComparison:
    """对比测试：验证与 PyTorch 的一致性"""
    
    def test_compare_sgd_with_pytorch(self):
        """对比 SGD 与 PyTorch 的结果"""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")
        
        # 设置随机种子
        np.random.seed(42)
        torch.manual_seed(42)
        
        # 创建相同的初始参数
        init_data = np.array([1.0, 2.0, 3.0])
        
        # Nanograd
        x_nano = Tensor(init_data.copy(), requires_grad=True)
        opt_nano = SGD([x_nano], lr=0.1)
        
        # PyTorch
        x_torch = torch.tensor(init_data.copy(), requires_grad=True)
        opt_torch = torch.optim.SGD([x_torch], lr=0.1)
        
        # 执行几步优化
        for i in range(5):
            # 设置相同的梯度
            grad = np.random.randn(3)
            
            x_nano.grad = grad.copy()
            opt_nano.step()
            
            x_torch.grad = torch.tensor(grad.copy())
            opt_torch.step()
        
        # 对比结果
        np.testing.assert_allclose(
            x_nano.data, 
            x_torch.detach().numpy(), 
            rtol=1e-5,
            err_msg="Nanograd SGD 与 PyTorch SGD 结果不一致"
        )
    
    def test_compare_adam_with_pytorch(self):
        """对比 Adam 与 PyTorch 的结果"""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")
        
        np.random.seed(42)
        torch.manual_seed(42)
        
        init_data = np.array([1.0, 2.0])
        
        # Nanograd
        x_nano = Tensor(init_data.copy(), requires_grad=True)
        opt_nano = Adam([x_nano], lr=0.01, betas=(0.9, 0.999), eps=1e-8)
        
        # PyTorch
        x_torch = torch.tensor(init_data.copy(), requires_grad=True)
        opt_torch = torch.optim.Adam([x_torch], lr=0.01, betas=(0.9, 0.999), eps=1e-8)
        
        # 执行几步优化
        for i in range(5):
            grad = np.random.randn(2)
            
            x_nano.grad = grad.copy()
            opt_nano.step()
            
            x_torch.grad = torch.tensor(grad.copy())
            opt_torch.step()
        
        # 对比结果
        np.testing.assert_allclose(
            x_nano.data, 
            x_torch.detach().numpy(), 
            rtol=1e-5,
            err_msg="Nanograd Adam 与 PyTorch Adam 结果不一致"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
