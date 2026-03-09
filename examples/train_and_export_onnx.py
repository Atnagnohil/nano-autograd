"""训练模型并导出为 ONNX，然后使用 ONNX Runtime 进行推理"""
import sys
sys.path.insert(0, '../src')

import numpy as np
from nanograd.tensor import Tensor
from nanograd.nn import Linear, ReLU, Sigmoid
from nanograd.optim import Adam
from nanograd.export_onnx import export_to_onnx


class XORNet:
    """XOR 网络"""
    def __init__(self):
        self.fc1 = Linear(2, 4)
        self.relu = ReLU()
        self.fc2 = Linear(4, 1)
        self.sigmoid = Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
    def parameters(self):
        return self.fc1.parameters() + self.fc2.parameters()


def train_xor_model():
    """训练 XOR 模型"""
    print("="*60)
    print("步骤 1: 训练 XOR 模型")
    print("="*60)
    
    # XOR 数据
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)
    
    # 创建模型
    model = XORNet()
    optimizer = Adam(model.parameters(), lr=0.01)
    
    # 训练
    print("\n训练中...")
    for epoch in range(2000):
        X_tensor = Tensor(X)
        pred = model.forward(X_tensor)
        
        # MSE 损失
        diff = pred.data - y
        loss_data = np.mean(diff ** 2)
        loss = Tensor(loss_data, _children=(pred,), _op='mse')
        
        def _backward():
            pred.grad += 2 * diff / len(y)
        loss._backward = _backward
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/2000, Loss: {loss_data:.6f}")
    
    # 测试
    print("\n训练完成！测试结果:")
    X_tensor = Tensor(X)
    pred = model.forward(X_tensor)
    
    for i in range(len(X)):
        pred_val = pred.data[i][0]
        true_val = y[i][0]
        pred_class = 1 if pred_val > 0.5 else 0
        true_class = int(true_val)
        status = "✅" if pred_class == true_class else "❌"
        print(f"{status} 输入: {X[i]}, 预测: {pred_val:.4f}, 真实: {true_val}")
    
    return model


def export_model(model):
    """导出模型为 ONNX"""
    print("\n" + "="*60)
    print("步骤 2: 导出模型为 ONNX")
    print("="*60)
    
    export_to_onnx(
        model,
        input_shape=(1, 2),  # 单个样本，2 个特征
        output_path="xor_trained.onnx",
        model_name="XORNet_Trained"
    )


def test_onnx_inference(model):
    """使用 ONNX Runtime 进行推理并对比"""
    print("\n" + "="*60)
    print("步骤 3: 使用 ONNX Runtime 推理")
    print("="*60)
    
    try:
        import onnxruntime as ort
    except ImportError:
        print("⚠️  未安装 onnxruntime")
        print("安装命令: pip install onnxruntime")
        return
    
    # 测试数据
    X_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    
    # Nanograd 推理
    print("\nNanograd 推理:")
    nano_outputs = []
    for x in X_test:
        x_tensor = Tensor(x.reshape(1, 2))
        output = model.forward(x_tensor)
        nano_outputs.append(output.data[0][0])
        print(f"  输入: {x}, 输出: {output.data[0][0]:.4f}")
    
    # ONNX Runtime 推理
    print("\nONNX Runtime 推理:")
    session = ort.InferenceSession("xor_trained.onnx")
    input_name = session.get_inputs()[0].name
    
    onnx_outputs = []
    for x in X_test:
        x_input = x.reshape(1, 2)
        output = session.run(None, {input_name: x_input})[0]
        onnx_outputs.append(output[0][0])
        print(f"  输入: {x}, 输出: {output[0][0]:.4f}")
    
    # 对比结果
    print("\n对比结果:")
    max_diff = 0
    for i, (nano, onnx) in enumerate(zip(nano_outputs, onnx_outputs)):
        diff = abs(nano - onnx)
        max_diff = max(max_diff, diff)
        print(f"  样本 {i}: Nanograd={nano:.6f}, ONNX={onnx:.6f}, 差异={diff:.8f}")
    
    print(f"\n最大差异: {max_diff:.8f}")
    
    if max_diff < 1e-5:
        print("✅ 验证通过！Nanograd 和 ONNX Runtime 输出完全一致")
    else:
        print(f"⚠️  存在差异: {max_diff}")


def benchmark_inference():
    """性能基准测试"""
    print("\n" + "="*60)
    print("步骤 4: 性能基准测试")
    print("="*60)
    
    try:
        import onnxruntime as ort
        import time
    except ImportError:
        print("⚠️  未安装 onnxruntime，跳过性能测试")
        return
    
    # 创建模型
    model = XORNet()
    
    # 测试数据
    X_test = np.random.randn(1000, 2).astype(np.float32)
    
    # Nanograd 推理
    print("\n测试 Nanograd 推理速度...")
    start = time.time()
    for x in X_test:
        x_tensor = Tensor(x.reshape(1, 2))
        _ = model.forward(x_tensor)
    nano_time = time.time() - start
    
    # ONNX Runtime 推理
    print("测试 ONNX Runtime 推理速度...")
    session = ort.InferenceSession("xor_trained.onnx")
    input_name = session.get_inputs()[0].name
    
    start = time.time()
    for x in X_test:
        x_input = x.reshape(1, 2)
        _ = session.run(None, {input_name: x_input})
    onnx_time = time.time() - start
    
    # 结果
    print(f"\n性能对比 (1000 次推理):")
    print(f"  Nanograd:     {nano_time:.3f}s ({1000/nano_time:.1f} samples/s)")
    print(f"  ONNX Runtime: {onnx_time:.3f}s ({1000/onnx_time:.1f} samples/s)")
    print(f"  加速比:       {nano_time/onnx_time:.2f}x")
    
    if onnx_time < nano_time:
        print(f"\n✅ ONNX Runtime 比 Nanograd 快 {nano_time/onnx_time:.2f} 倍")
    else:
        print(f"\n⚠️  Nanograd 比 ONNX Runtime 快 {onnx_time/nano_time:.2f} 倍")


def main():
    """主函数"""
    print("\n" + "="*60)
    print("训练、导出和推理完整流程")
    print("="*60)
    
    # 设置随机种子
    np.random.seed(42)
    
    # 1. 训练模型
    model = train_xor_model()
    
    # 2. 导出 ONNX
    export_model(model)
    
    # 3. ONNX 推理验证
    test_onnx_inference(model)
    
    # 4. 性能测试
    benchmark_inference()
    
    print("\n" + "="*60)
    print("完成！")
    print("="*60)
    print("\n生成的文件:")
    print("  - xor_trained.onnx (可以用 netron 可视化)")
    print("\n使用方法:")
    print("  1. 可视化: netron xor_trained.onnx")
    print("  2. 推理:")
    print("     import onnxruntime as ort")
    print("     session = ort.InferenceSession('xor_trained.onnx')")
    print("     output = session.run(None, {'tensor_0': [[0, 1]]})")
    print("="*60)


if __name__ == "__main__":
    main()
