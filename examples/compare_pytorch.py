"""与 PyTorch 对比验证梯度一致性和性能"""
import sys
sys.path.insert(0, '../src')

import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim

from nanograd.tensor import Tensor
from nanograd.nn import Linear, ReLU
from nanograd.optim import Adam


class NanoMLP:
    """Nanograd MLP"""
    def __init__(self, input_size, hidden_size, output_size):
        self.fc1 = Linear(input_size, hidden_size)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    def parameters(self):
        return self.fc1.parameters() + self.fc2.parameters()


class TorchMLP(nn.Module):
    """PyTorch MLP"""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def compare_gradients():
    """对比梯度计算的一致性"""
    print("=" * 60)
    print("测试 1: 梯度一致性验证")
    print("=" * 60)
    
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 创建相同的输入
    input_size, hidden_size, output_size = 10, 20, 5
    batch_size = 4
    
    X = np.random.randn(batch_size, input_size).astype(np.float32)
    
    # Nanograd 模型
    nano_model = NanoMLP(input_size, hidden_size, output_size)
    
    # PyTorch 模型
    torch_model = TorchMLP(input_size, hidden_size, output_size)
    
    # 复制权重（确保两个模型初始权重相同）
    with torch.no_grad():
        torch_model.fc1.weight.copy_(torch.from_numpy(nano_model.fc1.weight.data))
        torch_model.fc1.bias.copy_(torch.from_numpy(nano_model.fc1.bias.data))
        torch_model.fc2.weight.copy_(torch.from_numpy(nano_model.fc2.weight.data))
        torch_model.fc2.bias.copy_(torch.from_numpy(nano_model.fc2.bias.data))
    
    # 前向传播
    nano_x = Tensor(X)
    nano_out = nano_model.forward(nano_x)
    
    torch_x = torch.from_numpy(X)
    torch_out = torch_model(torch_x)
    
    # 对比前向传播结果
    print("\n前向传播对比:")
    print(f"Nanograd 输出形状: {nano_out.data.shape}")
    print(f"PyTorch 输出形状: {torch_out.shape}")
    print(f"输出差异 (max abs diff): {np.max(np.abs(nano_out.data - torch_out.detach().numpy())):.8f}")
    
    # 反向传播
    nano_loss = (nano_out * nano_out).sum()
    nano_loss.backward()
    
    torch_loss = (torch_out * torch_out).sum()
    torch_loss.backward()
    
    # 对比梯度
    print("\n梯度对比:")
    
    # fc1 权重梯度
    nano_fc1_grad = nano_model.fc1.weight.grad
    torch_fc1_grad = torch_model.fc1.weight.grad.numpy()
    diff_fc1 = np.max(np.abs(nano_fc1_grad - torch_fc1_grad))
    print(f"fc1.weight 梯度差异: {diff_fc1:.8f}")
    
    # fc1 偏置梯度
    nano_fc1_bias_grad = nano_model.fc1.bias.grad
    torch_fc1_bias_grad = torch_model.fc1.bias.grad.numpy()
    diff_fc1_bias = np.max(np.abs(nano_fc1_bias_grad - torch_fc1_bias_grad))
    print(f"fc1.bias 梯度差异: {diff_fc1_bias:.8f}")
    
    # fc2 权重梯度
    nano_fc2_grad = nano_model.fc2.weight.grad
    torch_fc2_grad = torch_model.fc2.weight.grad.numpy()
    diff_fc2 = np.max(np.abs(nano_fc2_grad - torch_fc2_grad))
    print(f"fc2.weight 梯度差异: {diff_fc2:.8f}")
    
    # fc2 偏置梯度
    nano_fc2_bias_grad = nano_model.fc2.bias.grad
    torch_fc2_bias_grad = torch_model.fc2.bias.grad.numpy()
    diff_fc2_bias = np.max(np.abs(nano_fc2_bias_grad - torch_fc2_bias_grad))
    print(f"fc2.bias 梯度差异: {diff_fc2_bias:.8f}")
    
    # 判断是否通过
    max_diff = max(diff_fc1, diff_fc1_bias, diff_fc2, diff_fc2_bias)
    if max_diff < 1e-5:
        print(f"\n✅ 梯度验证通过！最大差异: {max_diff:.8f}")
    else:
        print(f"\n❌ 梯度验证失败！最大差异: {max_diff:.8f}")


def compare_optimizer():
    """对比优化器更新的一致性"""
    print("\n" + "=" * 60)
    print("测试 2: 优化器一致性验证")
    print("=" * 60)
    
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 创建简单的参数
    init_data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    
    # Nanograd
    nano_param = Tensor(init_data.copy(), requires_grad=True)
    nano_opt = Adam([nano_param], lr=0.01)
    
    # PyTorch
    torch_param = torch.tensor(init_data.copy(), requires_grad=True)
    torch_opt = optim.Adam([torch_param], lr=0.01)
    
    print("\n执行 5 步优化...")
    for step in range(5):
        # 设置相同的梯度
        grad = np.random.randn(2, 2).astype(np.float32)
        
        # Nanograd
        nano_param.grad = grad.copy()
        nano_opt.step()
        
        # PyTorch
        torch_param.grad = torch.from_numpy(grad.copy())
        torch_opt.step()
        
        # 对比
        diff = np.max(np.abs(nano_param.data - torch_param.detach().numpy()))
        print(f"Step {step+1}: 参数差异 = {diff:.8f}")
    
    final_diff = np.max(np.abs(nano_param.data - torch_param.detach().numpy()))
    if final_diff < 1e-5:
        print(f"\n✅ 优化器验证通过！最终差异: {final_diff:.8f}")
    else:
        print(f"\n❌ 优化器验证失败！最终差异: {final_diff:.8f}")


def benchmark_performance():
    """性能基准测试"""
    print("\n" + "=" * 60)
    print("测试 3: 性能基准测试")
    print("=" * 60)
    
    input_size, hidden_size, output_size = 100, 200, 10
    batch_size = 32
    num_iterations = 100
    
    X = np.random.randn(batch_size, input_size).astype(np.float32)
    
    # Nanograd 基准测试
    print("\n测试 Nanograd...")
    nano_model = NanoMLP(input_size, hidden_size, output_size)
    nano_opt = Adam(nano_model.parameters(), lr=0.001)
    
    start_time = time.time()
    for _ in range(num_iterations):
        nano_x = Tensor(X)
        nano_out = nano_model.forward(nano_x)
        nano_loss = (nano_out * nano_out).sum()
        nano_opt.zero_grad()
        nano_loss.backward()
        nano_opt.step()
    nano_time = time.time() - start_time
    
    # PyTorch 基准测试
    print("测试 PyTorch...")
    torch_model = TorchMLP(input_size, hidden_size, output_size)
    torch_opt = optim.Adam(torch_model.parameters(), lr=0.001)
    
    start_time = time.time()
    for _ in range(num_iterations):
        torch_x = torch.from_numpy(X)
        torch_out = torch_model(torch_x)
        torch_loss = (torch_out * torch_out).sum()
        torch_opt.zero_grad()
        torch_loss.backward()
        torch_opt.step()
    torch_time = time.time() - start_time
    
    # 结果
    print(f"\n性能对比 ({num_iterations} 次迭代):")
    print(f"Nanograd: {nano_time:.3f}s")
    print(f"PyTorch:  {torch_time:.3f}s")
    print(f"速度比:   {nano_time/torch_time:.2f}x (Nanograd 相对于 PyTorch)")
    
    if nano_time / torch_time < 100:
        print("\n✅ 性能在合理范围内（纯 Python 实现预期会慢一些）")
    else:
        print("\n⚠️  性能差距较大，可能需要优化")


def main():
    """运行所有对比测试"""
    print("\n" + "=" * 60)
    print("Nanograd vs PyTorch 对比测试")
    print("=" * 60)
    
    try:
        compare_gradients()
        compare_optimizer()
        benchmark_performance()
        
        print("\n" + "=" * 60)
        print("所有测试完成！")
        print("=" * 60)
    
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
