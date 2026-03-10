# Nanograd

<div align="center">

**A lightweight autograd engine built from scratch for educational purposes**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)
[![ONNX](https://img.shields.io/badge/ONNX-supported-orange.svg)](https://onnx.ai/)
[![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet.svg)](https://github.com/astral-sh/uv)

*An educational deep learning framework for understanding automatic differentiation and neural network fundamentals*

*Powered by [uv](https://github.com/astral-sh/uv) package manager ⚡*

English | [简体中文](README_zh.md)

[Features](#features) • [Installation](#installation) • [Quick Start](#quick-start) • [How It Works](#how-it-works) • [Examples](#examples) • [Testing](#testing)

</div>

---


```python
# Scalar autograd
from nanograd.engine import Value

a = Value(2.0)
b = Value(3.0)
c = a * b + a ** 2
c.backward()
print(f"dc/da = {a.grad}")  # 7.0

# Neural networks
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

## Architecture

```
Input Data
    │
    ▼
Tensor (with grad tracking)
    │
    ▼
Computation Graph
    │
    ▼
Autograd Engine (topological sort + backprop)
    │
    ▼
Gradients → Optimizer → Updated Parameters
```

## Features

- **Automatic differentiation** for scalars and tensors
- **Neural network layers**: Linear, ReLU, Sigmoid, Tanh  
- **Optimizers**: SGD with momentum, Adam with bias correction
- **ONNX export** for deployment
- **PyTorch compatibility** - gradients match within 1e-5
- **Pure Python** - ~2000 lines, easy to understand

## Installation

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/Atnagnohil/nano-autograd.git
cd nano-autograd
uv sync

# Or use pip
pip install -e .
```

## Quick Start

Train XOR in 30 seconds:

```python
from nanograd.nn import MLP
from nanograd.optim import SGD

# Data
X = [[0,0], [0,1], [1,0], [1,1]]
y = [0, 1, 1, 0]

# Model
model = MLP([2, 4, 1])
optimizer = SGD(model.parameters(), lr=0.1)

# Train
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

# Test
for xi, yi in zip(X, y):
    pred = model(xi)
    print(f"Input: {xi}, Predicted: {pred.data:.4f}, Expected: {yi}")
```

## How It Works

This project shows how to build an autograd engine from scratch in 5 days:

### Day 1: Scalar Autograd Engine

Build the core `Value` class with automatic differentiation:

```python
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
    
    def backward(self):
        # Topological sort + backpropagation
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

### Day 2: Tensor Operations

Extend to multi-dimensional arrays with broadcasting:

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

### Day 3: Neural Network Layers

Build composable modules:

```python
class Linear(Module):
    def __init__(self, in_features, out_features):
        # Kaiming initialization
        bound = np.sqrt(6.0 / in_features)
        self.weight = Tensor(
            np.random.uniform(-bound, bound, (out_features, in_features)),
            requires_grad=True
        )
        self.bias = Tensor(np.zeros(out_features), requires_grad=True)
    
    def forward(self, x):
        return x @ self.weight.T + self.bias
```

### Day 4: Optimizers

Implement parameter update strategies:

```python
class Adam(Optimizer):
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            # Update biased first and second moment estimates
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * p.grad ** 2
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

### Day 5: ONNX Export

Deploy models to production:

```python
class ONNXExporter:
    def _convert_op(self, tensor):
        if tensor._op == '@':
            node = helper.make_node('MatMul', input_names, [output_name])
        elif tensor._op == 'relu':
            node = helper.make_node('Relu', [input_name], [output_name])
        self.nodes.append(node)
```

## Examples

### XOR Problem
```bash
cd examples
python train_xor.py
```

Output:
```
Epoch 0, Loss: 4.0000
Epoch 100, Loss: 0.0001
✅ [0, 0] → 0.0001 (expected: 0)
✅ [0, 1] → 1.0000 (expected: 1)
✅ [1, 0] → 1.0000 (expected: 1)  
✅ [1, 1] → 0.0000 (expected: 0)
```

### MNIST Training
```bash
python train_mlp.py
```

### PyTorch Comparison
```bash
python compare_pytorch.py
```

Output:
```
✅ Gradient verification passed! Max difference: 0.00000381
✅ Optimizer verification passed! Final difference: 0.00000000
```

## Project Structure

```
src/nanograd/
├── engine.py          # Scalar autograd
├── tensor.py          # Tensor class  
├── ops.py             # Tensor operations
├── nn/
│   ├── module.py      # Base Module class
│   ├── linear.py      # Linear layer
│   └── activation.py  # Activation functions
└── optim/
    ├── sgd.py         # SGD optimizer
    └── adam.py        # Adam optimizer

examples/
├── train_xor.py       # XOR problem
├── train_mlp.py       # MNIST training
└── compare_pytorch.py # PyTorch comparison

tests/                 # Unit tests
```

## Testing

```bash
uv run pytest tests/ -v
```

All tests pass with gradients matching PyTorch within 1e-5.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Inspiration

This project was inspired by:
- [micrograd](https://github.com/karpathy/micrograd) by Andrej Karpathy
- [tinygrad](https://github.com/tinygrad/tinygrad) by George Hotz  
- CS231n course materials
