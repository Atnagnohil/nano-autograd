# Nanograd 示例

[English](README.md) | 简体中文

使用 Nanograd 框架的完整示例，从简单的 XOR 问题到 MNIST 分类，再到 ONNX 模型导出。

## 快速开始

### 1. XOR 问题训练

最简单的示例，验证框架的基本功能。

```bash
python train_xor.py
```

**特点:**
- 训练时间: ~5 秒
- 网络结构: 2 → 4 (ReLU) → 1 (Sigmoid)
- 准确率: 100%

**输出:**
```
输入: [0. 0.], 预测: 0.0001, 真实: 0.0  ✅
输入: [0. 1.], 预测: 1.0000, 真实: 1.0  ✅
输入: [1. 0.], 预测: 1.0000, 真实: 1.0  ✅
输入: [1. 1.], 预测: 0.0000, 真实: 0.0  ✅
```

### 2. MNIST 训练

在 MNIST 手写数字数据集上训练多层感知机。

```bash
python train_mlp.py
```

**网络结构:**
- 输入: 784 (28×28 图像)
- 隐藏层 1: 128 + ReLU
- 隐藏层 2: 64 + ReLU
- 输出: 10 类别

**训练配置:**
- 优化器: Adam (lr=0.001)
- 批次大小: 32
- 训练轮数: 10

### 3. PyTorch 对比验证

全面验证 Nanograd 与 PyTorch 的一致性。

```bash
python compare_pytorch.py
```

**测试内容:**

1. **梯度一致性** ✅
   - 前向传播输出差异: < 1e-6
   - 反向传播梯度差异: < 1e-5

2. **优化器一致性** ✅
   - Adam 参数更新差异: 0

3. **性能基准测试** ✅
   - 100 次迭代对比
   - Nanograd 性能在合理范围内

### 4. ONNX 导出

将训练好的模型导出为 ONNX 格式。

#### 4.1 基础导出示例

```bash
python export_onnx_example.py
```

导出两个模型:
- `simple_mlp.onnx` - 简单的 MLP
- `xor_net.onnx` - XOR 网络

#### 4.2 完整训练和导出流程

```bash
python train_and_export_onnx.py
```

**流程:**
1. 训练 XOR 模型（2000 epochs）
2. 导出为 ONNX 格式
3. 使用 ONNX Runtime 验证
4. 性能基准测试

**输出:**
- `xor_trained.onnx` - 训练好的模型
- 推理结果对比
- 性能测试报告

## ONNX 导出详细指南

### 支持的操作

| 类别 | 操作 | ONNX 算子 |
|------|------|-----------|
| 基础运算 | Add, Sub, Mul, Div | Add, Sub, Mul, Div |
| 矩阵运算 | MatMul | MatMul |
| 激活函数 | ReLU, Sigmoid, Tanh | Relu, Sigmoid, Tanh |
| 形状操作 | Reshape, Transpose | Reshape, Transpose |
| 聚合操作 | Sum, Mean | ReduceSum, ReduceMean |

### 快速导出

```python
from nanograd.nn import Linear, ReLU
from nanograd import export_to_onnx

# 定义模型
class MyModel:
    def __init__(self):
        self.fc1 = Linear(10, 20)
        self.relu = ReLU()
        self.fc2 = Linear(20, 5)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    
    def parameters(self):
        return self.fc1.parameters() + self.fc2.parameters()

# 导出
model = MyModel()
export_to_onnx(model, input_shape=(1, 10), output_path="model.onnx")
```

### 使用 ONNX Runtime 推理

```python
import onnxruntime as ort
import numpy as np

# 加载模型
session = ort.InferenceSession("model.onnx")

# 推理
input_name = session.get_inputs()[0].name
input_data = np.random.randn(1, 10).astype(np.float32)
output = session.run(None, {input_name: input_data})
print(output[0])
```

### 可视化模型

```bash
# 安装 Netron
pip install netron

# 可视化
netron model.onnx
```

## 使用场景

### 场景 1: 学习自动微分

从 `train_xor.py` 开始，理解:
- 前向传播
- 反向传播
- 梯度下降

### 场景 2: 训练神经网络

使用 `train_mlp.py` 学习:
- 数据加载和预处理
- 批量训练
- 损失函数
- 模型评估

### 场景 3: 验证实现正确性

运行 `compare_pytorch.py` 确保:
- 梯度计算正确
- 优化器实现正确
- 性能可接受

### 场景 4: 模型部署

使用 ONNX 导出实现:
- 跨平台部署
- 生产环境推理
- 硬件加速

## 故障排除

### 问题 1: 找不到 MNIST 数据

**解决方案**: 脚本会自动使用模拟数据，或下载真实数据到 `Data/minist/`

### 问题 2: ONNX 导出失败

**检查**:
- 是否安装了 `onnx`: `pip install onnx`
- 模型是否有 `forward()` 和 `parameters()` 方法

### 问题 3: ONNX Runtime 推理失败

**检查**:
- 是否安装了 `onnxruntime`: `pip install onnxruntime`
- 输入形状是否匹配
- 数据类型是否为 `float32`

## 学习路径

1. **第 1 天**: 运行 `train_xor.py`，理解基本概念
2. **第 2 天**: 运行 `train_mlp.py`，学习完整训练流程
3. **第 3 天**: 运行 `compare_pytorch.py`，验证实现
4. **第 4 天**: 学习 ONNX 导出，运行 `export_onnx_example.py`
5. **第 5 天**: 完整流程，运行 `train_and_export_onnx.py`

## 扩展练习

1. **修改 XOR 网络**: 尝试不同的隐藏层大小
2. **添加新的激活函数**: 实现 LeakyReLU 或 ELU
3. **实现新的优化器**: 尝试 RMSprop 或 AdaGrad
4. **训练其他数据集**: Fashion-MNIST 或 CIFAR-10
5. **导出更复杂的模型**: CNN 或 RNN

## 参考资源

- [Nanograd 主文档](../README.md)
- [ONNX 官方文档](https://onnx.ai/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Netron 可视化](https://netron.app/)

## 提示

- 所有示例都使用相对路径，可以直接运行
- 生成的图片和模型文件保存在当前目录
- 使用 `python -u` 可以实时查看输出
- 建议在虚拟环境中运行

Happy coding! 🚀
