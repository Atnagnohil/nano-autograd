# Nanograd Examples

English | [简体中文](README_zh.md)

Complete examples using the Nanograd framework, from simple XOR problems to MNIST classification and ONNX model export.

## Quick Start

### 1. XOR Problem Training

The simplest example to verify basic framework functionality.

```bash
python train_xor.py
```

**Features:**
- Training time: ~5 seconds
- Network structure: 2 → 4 (ReLU) → 1 (Sigmoid)
- Accuracy: 100%

**Output:**
```
Input: [0. 0.], Predicted: 0.0001, Expected: 0.0  ✅
Input: [0. 1.], Predicted: 1.0000, Expected: 1.0  ✅
Input: [1. 0.], Predicted: 1.0000, Expected: 1.0  ✅
Input: [1. 1.], Predicted: 0.0000, Expected: 0.0  ✅
```

### 2. MNIST Training

Train a multi-layer perceptron on the MNIST handwritten digit dataset.

```bash
python train_mlp.py
```

**Network Structure:**
- Input: 784 (28×28 images)
- Hidden layer 1: 128 + ReLU
- Hidden layer 2: 64 + ReLU
- Output: 10 classes

**Training Configuration:**
- Optimizer: Adam (lr=0.001)
- Batch size: 32
- Epochs: 10

### 3. PyTorch Comparison

Comprehensive validation of Nanograd's consistency with PyTorch.

```bash
python compare_pytorch.py
```

**Test Content:**

1. **Gradient Consistency** ✅
   - Forward propagation output difference: < 1e-6
   - Backward propagation gradient difference: < 1e-5

2. **Optimizer Consistency** ✅
   - Adam parameter update difference: 0

3. **Performance Benchmark** ✅
   - 100 iteration comparison
   - Nanograd performance within reasonable range

### 4. ONNX Export

Export trained models to ONNX format.

#### 4.1 Basic Export Example

```bash
python export_onnx_example.py
```

Exports two models:
- `simple_mlp.onnx` - Simple MLP
- `xor_net.onnx` - XOR network

#### 4.2 Complete Training and Export Pipeline

```bash
python train_and_export_onnx.py
```

**Pipeline:**
1. Train XOR model (2000 epochs)
2. Export to ONNX format
3. Validate with ONNX Runtime
4. Performance benchmark

**Output:**
- `xor_trained.onnx` - Trained model
- Inference result comparison
- Performance test report

## ONNX Export Guide

### Supported Operations

| Category | Operations | ONNX Operators |
|----------|-----------|----------------|
| Basic Ops | Add, Sub, Mul, Div | Add, Sub, Mul, Div |
| Matrix Ops | MatMul | MatMul |
| Activations | ReLU, Sigmoid, Tanh | Relu, Sigmoid, Tanh |
| Shape Ops | Reshape, Transpose | Reshape, Transpose |
| Reduction | Sum, Mean | ReduceSum, ReduceMean |

### Quick Export

```python
from nanograd.nn import Linear, ReLU
from nanograd import export_to_onnx

# Define model
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

# Export
model = MyModel()
export_to_onnx(model, input_shape=(1, 10), output_path="model.onnx")
```

### ONNX Runtime Inference

```python
import onnxruntime as ort
import numpy as np

# Load model
session = ort.InferenceSession("model.onnx")

# Inference
input_name = session.get_inputs()[0].name
input_data = np.random.randn(1, 10).astype(np.float32)
output = session.run(None, {input_name: input_data})
print(output[0])
```

### Visualize Model

```bash
# Install Netron
pip install netron

# Visualize
netron model.onnx
```

## Use Cases

### Case 1: Learning Automatic Differentiation

Start with `train_xor.py` to understand:
- Forward propagation
- Backward propagation
- Gradient descent

### Case 2: Training Neural Networks

Use `train_mlp.py` to learn:
- Data loading and preprocessing
- Batch training
- Loss functions
- Model evaluation

### Case 3: Verify Implementation Correctness

Run `compare_pytorch.py` to ensure:
- Gradient computation is correct
- Optimizer implementation is correct
- Performance is acceptable

### Case 4: Model Deployment

Use ONNX export for:
- Cross-platform deployment
- Production environment inference
- Hardware acceleration

## Troubleshooting

### Issue 1: MNIST Data Not Found

**Solution**: Scripts automatically use simulated data, or download real data to `Data/minist/`

### Issue 2: ONNX Export Failed

**Check**:
- Is `onnx` installed: `pip install onnx`
- Does model have `forward()` and `parameters()` methods

### Issue 3: ONNX Runtime Inference Failed

**Check**:
- Is `onnxruntime` installed: `pip install onnxruntime`
- Does input shape match
- Is data type `float32`

## Learning Path

1. **Day 1**: Run `train_xor.py`, understand basic concepts
2. **Day 2**: Run `train_mlp.py`, learn complete training pipeline
3. **Day 3**: Run `compare_pytorch.py`, verify implementation
4. **Day 4**: Learn ONNX export, run `export_onnx_example.py`
5. **Day 5**: Complete pipeline, run `train_and_export_onnx.py`

## Extension Exercises

1. **Modify XOR Network**: Try different hidden layer sizes
2. **Add New Activation Functions**: Implement LeakyReLU or ELU
3. **Implement New Optimizers**: Try RMSprop or AdaGrad
4. **Train Other Datasets**: Fashion-MNIST or CIFAR-10
5. **Export More Complex Models**: CNN or RNN

## References

- [Nanograd Main Documentation](../README.md)
- [ONNX Official Documentation](https://onnx.ai/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Netron Visualization](https://netron.app/)

## Tips

- All examples use relative paths and can be run directly
- Generated images and model files are saved in the current directory
- Use `python -u` to view output in real-time
- Recommended to run in a virtual environment

Happy coding! 🚀
