# Nano Autograd

从0开始实现一个简易的自动微分框架

## 开发路线图

### 阶段 1: 核心自动微分引擎 ✅
**目标**: 实现基础的计算图和反向传播机制

- [ ] `engine.py` - 实现 `Value` 类（标量自动微分）
  - 支持基本运算：加、减、乘、除、幂
  - 实现 `backward()` 方法（反向传播）
  - 构建计算图（拓扑排序）
- [ ] `tests/test_engine.py` - 验证标量梯度计算正确性
  - 与 PyTorch 对比验证

### 阶段 2: 张量运算支持 🚧
**目标**: 从标量扩展到多维张量

- [ ] `tensor.py` - 实现 `Tensor` 类
  - 基于 NumPy 的多维数组
  - 支持广播机制
  - 梯度累积和存储
- [ ] `ops.py` - 实现张量运算
  - 矩阵乘法 (matmul)
  - 元素级运算 (add, mul, pow)
  - Reshape, transpose, sum, mean
- [ ] `tests/test_tensor.py` - 验证张量梯度

### 阶段 3: 神经网络层 📦
**目标**: 构建可组合的神经网络模块

- [ ] `nn/__init__.py` - 模块基类
- [ ] `nn/linear.py` - 全连接层
  - 权重初始化（Xavier/He）
  - 前向传播和参数管理
- [ ] `nn/activation.py` - 激活函数
  - ReLU, Sigmoid, Tanh
  - Softmax（用于分类）
- [ ] `tests/test_nn.py` - 验证层的前向和反向传播

### 阶段 4: 优化器 🎯
**目标**: 实现参数更新策略

- [ ] `optim/__init__.py` - 优化器基类
- [ ] `optim/sgd.py` - 随机梯度下降
  - 支持学习率
  - 支持动量（可选）
- [ ] `optim/adam.py` - Adam 优化器（可选）
- [ ] `tests/test_optim.py` - 验证参数更新

### 阶段 5: 端到端示例 🚀
**目标**: 训练一个完整的神经网络

- [ ] `examples/train_mlp.py` - 多层感知机训练
  - 使用经典数据集（如 XOR 或 MNIST 简化版）
  - 完整的训练循环
  - 损失曲线可视化
- [ ] `examples/compare_pytorch.py` - 与 PyTorch 对比
  - 验证梯度一致性
  - 性能基准测试

### 阶段 6: 文档和优化 📚
**目标**: 完善项目质量

- [ ] 添加详细的 API 文档
- [ ] 完善支持更多基础算子
- [ ] 性能分析和优化
- [ ] 添加更多测试用例
- [ ] 编写使用教程

---

## 项目结构
```txt
nano_autograd/
│
├── src/
│   └── nano_autograd/
│       │
│       ├── __init__.py
│       │
│       ├── tensor.py
│       ├── engine.py
│       ├── ops.py
│       │
│       ├── nn/
│       │   ├── __init__.py
│       │   ├── linear.py
│       │   └── activation.py
│       │
│       └── optim/
│           ├── __init__.py
│           └── sgd.py
│
├── examples/
│   └── train_mlp.py
│
├── tests/
│   └── test_autograd.py
│
├── pyproject.toml
|——uv.lock
└── README.md
```

## 技术栈

- **包管理**: `uv` - 快速的 Python 包管理器
- **核心依赖**: `numpy` - 数值计算
- **开发依赖**: 
  - `pytest` - 单元测试
  - `torch` (CPU) - 用于验证梯度正确性
  - `ruff` - 代码格式化和检查

## 快速开始

### 安装依赖

```bash
# 安装 uv（如果还没有）
pip install uv

# 创建虚拟环境并安装依赖
uv sync

# 或者手动安装
uv pip install -e .[dev]
```

### 验证安装

```bash
# 验证 PyTorch 安装
python tests/verify_torch.py

# 运行测试（开发过程中）
pytest tests/
```

## 参考项目
https://github.com/srkds/Micrograd-Autograd-Engine-implementation
https://github.com/tinygrad/tinygrad
https://github.com/karpathy/micrograd/

## 开发指南

### 当前进度
- ✅ 项目结构搭建
- 🚧 正在开发：核心自动微分引擎

### 下一步
从 **阶段 1** 开始，实现 `engine.py` 中的标量自动微分。

## 学习资源

- [Andrej Karpathy - micrograd](https://github.com/karpathy/micrograd)
- [PyTorch Autograd 文档](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)
- [CS231n - Backpropagation](http://cs231n.github.io/optimization-2/)

## 许可证

MIT