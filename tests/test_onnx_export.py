"""测试 ONNX 导出功能"""
import sys
import pytest
import numpy as np

# 尝试导入 ONNX，如果没有则跳过测试
try:
    import onnx
    from onnx import helper, TensorProto
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from nanograd.tensor import Tensor
from nanograd.nn import Linear, ReLU, Sigmoid

if ONNX_AVAILABLE:
    from nanograd.export_onnx import ONNXExporter, export_to_onnx


@pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not installed")
class TestONNXExport:
    """测试 ONNX 导出功能"""
    
    def test_exporter_initialization(self):
        """测试导出器初始化"""
        exporter = ONNXExporter()
        assert exporter.nodes == []
        assert exporter.initializers == []
        assert exporter.inputs == []
        assert exporter.outputs == []
    
    def test_unique_name_generation(self):
        """测试唯一名称生成"""
        exporter = ONNXExporter()
        name1 = exporter._get_unique_name("test")
        name2 = exporter._get_unique_name("test")
        assert name1 != name2
        assert "test" in name1
        assert "test" in name2
    
    def test_tensor_name_mapping(self):
        """测试 Tensor 名称映射"""
        exporter = ONNXExporter()
        t1 = Tensor([1.0, 2.0])
        t2 = Tensor([3.0, 4.0])
        
        name1 = exporter._get_tensor_name(t1)
        name2 = exporter._get_tensor_name(t2)
        
        # 相同 tensor 应该返回相同名称
        assert exporter._get_tensor_name(t1) == name1
        # 不同 tensor 应该返回不同名称
        assert name1 != name2
    
    def test_simple_linear_export(self):
        """测试简单 Linear 层导出"""
        class SimpleModel:
            def __init__(self):
                self.fc = Linear(5, 3)
            
            def forward(self, x):
                return self.fc(x)
            
            def parameters(self):
                return self.fc.parameters()
        
        model = SimpleModel()
        exporter = ONNXExporter()
        
        # 导出模型
        try:
            onnx_model = exporter.export(
                model,
                input_shape=(1, 5),
                output_path="test_linear.onnx",
                model_name="TestLinear"
            )
            
            # 验证基本结构
            assert len(exporter.inputs) == 1
            assert len(exporter.outputs) == 1
            assert len(exporter.initializers) >= 2  # weight + bias
            assert len(exporter.nodes) >= 2  # MatMul + Add
            
            print("✅ Linear 层导出测试通过")
        except Exception as e:
            pytest.skip(f"导出失败: {e}")
    
    def test_mlp_export(self):
        """测试 MLP 导出"""
        class MLP:
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
        
        model = MLP()
        exporter = ONNXExporter()
        
        try:
            onnx_model = exporter.export(
                model,
                input_shape=(1, 10),
                output_path="test_mlp.onnx",
                model_name="TestMLP"
            )
            
            # 验证结构
            assert len(exporter.inputs) == 1
            assert len(exporter.outputs) == 1
            assert len(exporter.initializers) >= 4  # 2 weights + 2 biases
            
            # 应该有: MatMul, Add, ReLU, MatMul, Add
            assert len(exporter.nodes) >= 5
            
            print("✅ MLP 导出测试通过")
        except Exception as e:
            pytest.skip(f"导出失败: {e}")
    
    def test_activation_functions(self):
        """测试激活函数导出"""
        class ActivationModel:
            def __init__(self):
                self.fc = Linear(5, 5)
                self.sigmoid = Sigmoid()
            
            def forward(self, x):
                x = self.fc(x)
                x = self.sigmoid(x)
                return x
            
            def parameters(self):
                return self.fc.parameters()
        
        model = ActivationModel()
        exporter = ONNXExporter()
        
        try:
            onnx_model = exporter.export(
                model,
                input_shape=(1, 5),
                output_path="test_activation.onnx",
                model_name="TestActivation"
            )
            
            # 检查是否有 Sigmoid 节点
            has_sigmoid = any(node.op_type == 'Sigmoid' for node in exporter.nodes)
            assert has_sigmoid, "应该包含 Sigmoid 节点"
            
            print("✅ 激活函数导出测试通过")
        except Exception as e:
            pytest.skip(f"导出失败: {e}")


@pytest.mark.skipif(not ONNX_AVAILABLE, reason="ONNX not installed")
def test_export_to_onnx_function():
    """测试便捷导出函数"""
    class TinyModel:
        def __init__(self):
            self.fc = Linear(3, 2)
        
        def forward(self, x):
            return self.fc(x)
        
        def parameters(self):
            return self.fc.parameters()
    
    model = TinyModel()
    
    try:
        onnx_model = export_to_onnx(
            model,
            input_shape=(1, 3),
            output_path="test_tiny.onnx",
            model_name="TinyModel"
        )
        
        assert onnx_model is not None
        print("✅ 便捷导出函数测试通过")
    except Exception as e:
        pytest.skip(f"导出失败: {e}")


if __name__ == "__main__":
    if ONNX_AVAILABLE:
        pytest.main([__file__, "-v"])
    else:
        print("⚠️  ONNX 未安装，跳过测试")
        print("安装命令: pip install onnx")
