"""演示如何将 Nanograd 模型导出为 ONNX 格式"""
import sys
sys.path.insert(0, '../src')

import numpy as np
from nanograd.tensor import Tensor
from nanograd.nn import Linear, ReLU, Sigmoid
from nanograd.export_onnx import export_to_onnx


class SimpleMLP:
    """简单的多层感知机"""
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


def test_onnx_export():
    """测试 ONNX 导出功能"""
    print("="*60)
    print("测试 ONNX 导出功能")
    print("="*60)
    
    # 测试 1: 简单 MLP
    print("\n1. 导出简单 MLP...")
    model1 = SimpleMLP()
    export_to_onnx(
        model1, 
        input_shape=(1, 10),  # batch_size=1, features=10
        output_path="simple_mlp.onnx",
        model_name="SimpleMLP"
    )
    
    # 测试 2: XOR 网络
    print("\n2. 导出 XOR 网络...")
    model2 = XORNet()
    export_to_onnx(
        model2,
        input_shape=(1, 2),  # batch_size=1, features=2
        output_path="xor_net.onnx",
        model_name="XORNet"
    )
    
    print("\n" + "="*60)
    print("导出完成！")
    print("="*60)


def verify_onnx_model():
    """验证导出的 ONNX 模型"""
    print("\n" + "="*60)
    print("验证 ONNX 模型")
    print("="*60)
    
    try:
        import onnxruntime as ort
    except ImportError:
        print("⚠️  未安装 onnxruntime，跳过验证")
        print("安装命令: pip install onnxruntime")
        return
    
    # 验证 SimpleMLP
    print("\n验证 SimpleMLP...")
    
    # 创建 Nanograd 模型
    nano_model = SimpleMLP()
    
    # 创建测试输入
    test_input = np.random.randn(1, 10).astype(np.float32)
    
    # Nanograd 推理
    nano_input = Tensor(test_input)
    nano_output = nano_model.forward(nano_input)
    
    # 导出 ONNX
    export_to_onnx(nano_model, (1, 10), "simple_mlp_verify.onnx")
    
    # ONNX Runtime 推理
    session = ort.InferenceSession("simple_mlp_verify.onnx")
    input_name = session.get_inputs()[0].name
    onnx_output = session.run(None, {input_name: test_input})[0]
    
    # 对比结果
    diff = np.max(np.abs(nano_output.data - onnx_output))
    print(f"\nNanograd 输出形状: {nano_output.shape}")
    print(f"ONNX 输出形状: {onnx_output.shape}")
    print(f"最大差异: {diff:.8f}")
    
    if diff < 1e-5:
        print("✅ 验证通过！Nanograd 和 ONNX 输出一致")
    else:
        print(f"⚠️  差异较大: {diff}")


def visualize_onnx_model():
    """可视化 ONNX 模型（可选）"""
    print("\n" + "="*60)
    print("可视化 ONNX 模型")
    print("="*60)
    
    try:
        import netron
        print("\n启动 Netron 可视化...")
        print("提示: 在浏览器中打开 http://localhost:8080")
        netron.start("simple_mlp.onnx")
    except ImportError:
        print("⚠️  未安装 netron，跳过可视化")
        print("安装命令: pip install netron")
        print("然后运行: netron simple_mlp.onnx")


if __name__ == "__main__":
    # 测试导出
    test_onnx_export()
    
    # 验证模型
    verify_onnx_model()
    
    # 可视化（可选）
    # visualize_onnx_model()
    
    print("\n" + "="*60)
    print("使用说明:")
    print("="*60)
    print("1. 查看 ONNX 模型: netron simple_mlp.onnx")
    print("2. 使用 ONNX Runtime 推理:")
    print("   import onnxruntime as ort")
    print("   session = ort.InferenceSession('simple_mlp.onnx')")
    print("   output = session.run(None, {'input': test_data})")
    print("="*60)
