# Nanograd

<div align="center">

**从零开始实现的轻量级自动微分框架**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)
[![ONNX](https://img.shields.io/badge/ONNX-supported-orange.svg)](https://onnx.ai/)
[![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet.svg)](https://github.com/astral-sh/uv)

*一个教育性的深度学习框架，用于理解自动微分和神经网络的核心原理*

*使用 [uv](https://github.com/astral-sh/uv) 包管理器 ⚡*

[English](README.md) | 简体中文

[特性](#特性) • [安装](#安装) • [快速开始](#快速开始) • [工作原理](#工作原理) • [示例](#示例) • [测试](#测试)

</div>

---

```python
# 标量自动微分
from nanograd.engine import Value

a = Value(2.0)
b = Value(3.0)
c = a * b + a ** 2
c.backward()
print(f"dc/da = {a.grad}")  # 7.0

# 神经网络
from nanograd.nn import Linear, ReLU
from nanograd.optim import Adam

model = MLP([784, 128, 10])
optimizer = Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    loss = model(x_train, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## 架构

```
输入数据
    │
    ▼
张量（梯度追踪）
    │
    ▼
计算图
    │
    ▼
自动微分引擎（拓扑排序 + 反向传播）
    │
    ▼
梯度 → 优化器 → 更新参数
```

## 特性

- **自动微分** - 支持标量和张量
- **神经网络层** - Linear, ReLU, Sigmoid, Tanh  
- **优化器** - 带动量的 SGD，带偏差修正的 Adam
- **ONNX 导出** - 用于部署
- **PyTorch 兼容** - 梯度误差在 1e-5 以内
- **纯 Python** - 约 2000 行代码，易于理解

## 安装

```bash
# 安装 uv（如果还没有）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆并安装
git clone https://github.com/yourusername/nano_autograd.git
cd nano_autograd
uv sync

# 或使用 pip
pip install -e .
```

## 快速开始

30 秒训练 XOR：

```python
from nanograd.nn import MLP
from nanograd.optim import SGD

# 数据
X = [[0,0], [0,1], [1,0], [1,1]]
y = [0, 1, 1, 0]

# 模型
model = MLP([2, 4, 1])
optimizer = SGD(model.parameters(), lr=0.1)

# 训练
for epoch in range(1000):
    total_loss = 0
    for xi, yi in zip(X, y):
        pred = model(xi)
        loss = (pred - yi) ** 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.data
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

# 测试
for xi, yi in zip(X, y):
    pred = model(xi)
    print(f"输入: {xi}, 预测: {pred.data:.4f}, 期望: {yi}")
```

## 工作原理

本项目展示如何在 5 天内从零构建自动微分引擎：

### 第 1 天：标量自动微分引擎

构建核心 `Value` 类，实现自动微分：

```python
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
    
    def backward(self):
        # 拓扑排序 + 反向传播
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
```

### 第 2 天：张量运算

扩展到支持广播的多维数组：

```python
def matmul(a, b):
    out = Tensor(a.data @ b.data, (a, b), '@')
    
    def _backward():
        if a.requires_grad:
            a.grad += out.grad @ b.data.T
        if b.requires_grad:
            b.grad += a.data.T @ out.grad
    
    out._backward = _backward
    return out
```

### 第 3 天：神经网络层

构建可组合的模块：

```python
class Linear(Module):
    def __init__(self, in_features, out_features):
        # Kaiming 初始化
        bound = np.sqrt(6.0 / in_features)
        self.weight = Tensor(
            np.random.uniform(-bound, bound, (out_features, in_features)),
            requires_grad=True
        )
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)
    
    def forward(self, x):
        return x @ self.weight.T + self.bias
```

### 第 4 天：优化器

实现参数更新策略：

```python
class Adam(Optimizer):
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            # 更新一阶和二阶矩估计
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * p.grad ** 2
            
            # 偏差修正
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # 更新参数
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

### 第 5 天：ONNX 导出

将模型部署到生产环境：

```python
class ONNXExporter:
    def _convert_op(self, tensor):
        if tensor._op == '@':
            node = helper.make_node('MatMul', input_names, [output_name])
        elif tensor._op == 'relu':
            node = helper.make_node('Relu', [input_name], [output_name])
        self.nodes.append(node)
```

## 示例

### XOR 问题
```bash
cd examples
python train_xor.py
```

输出：
```
Epoch 0, Loss: 4.0000
Epoch 100, Loss: 0.0001
✅ [0, 0] → 0.0001 (expected: 0)
✅ [0, 1] → 1.0000 (expected: 1)
✅ [1, 0] → 1.0000 (expected: 1)  
✅ [1, 1] → 0.0000 (expected: 0)
```

### MNIST 训练
```bash
python train_mlp.py
```

### PyTorch 对比
```bash
python compare_pytorch.py
```

输出：
```
✅ 梯度验证通过！最大差异: 0.00000381
✅ 优化器验证通过！最终差异: 0.00000000
```

## 项目结构

```
src/nanograd/
├── engine.py          # 标量自动微分
├── tensor.py          # 张量类
├── ops.py             # 张量运算
├── nn/
│   ├── module.py      # Module 基类
│   ├── linear.py      # 全连接层
│   └── activation.py  # 激活函数
└── optim/
    ├── sgd.py         # SGD 优化器
    └── adam.py        # Adam 优化器

examples/
├── train_xor.py       # XOR 问题
├── train_mlp.py       # MNIST 训练
└── compare_pytorch.py # PyTorch 对比

tests/                 # 单元测试
```

## 测试

```bash
uv run pytest tests/ -v
```

所有测试通过，梯度与 PyTorch 的误差在 1e-5 以内。

## 开源协议

MIT License - 详见 [LICENSE](LICENSE) 文件

## 致谢

本项目受以下项目启发：
- [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy
- [tinygrad](https://github.com/tinygrad/tinygrad) by George Hotz  
- CS231n 课程
