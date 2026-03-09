"""ONNX 导出功能"""
import numpy as np
from typing import List, Dict, Tuple, Any
import onnx
from onnx import helper, TensorProto, numpy_helper
from .tensor import Tensor


class ONNXExporter:
    """将 Nanograd 模型导出为 ONNX 格式"""
    
    def __init__(self):
        self.nodes = []  # ONNX 节点列表
        self.initializers = []  # 权重和常量
        self.inputs = []  # 输入
        self.outputs = []  # 输出
        self.value_info = []  # 中间值信息
        self.tensor_map = {}  # Tensor -> ONNX 名称映射
        self.name_counter = 0  # 用于生成唯一名称
    
    def _get_unique_name(self, prefix="tensor"):
        """生成唯一的张量名称"""
        name = f"{prefix}_{self.name_counter}"
        self.name_counter += 1
        return name
    
    def _get_tensor_name(self, tensor: Tensor) -> str:
        """获取或创建 Tensor 的 ONNX 名称"""
        if id(tensor) not in self.tensor_map:
            self.tensor_map[id(tensor)] = self._get_unique_name()
        return self.tensor_map[id(tensor)]
    
    def _numpy_dtype_to_onnx(self, np_dtype) -> int:
        """将 NumPy dtype 转换为 ONNX TensorProto 类型"""
        dtype_map = {
            np.float32: TensorProto.FLOAT,
            np.float64: TensorProto.DOUBLE,
            np.int32: TensorProto.INT32,
            np.int64: TensorProto.INT64,
        }
        return dtype_map.get(np_dtype, TensorProto.FLOAT)
    
    def _add_initializer(self, tensor: Tensor, name: str):
        """添加权重或常量到 initializers"""
        onnx_tensor = numpy_helper.from_array(tensor.data, name)
        self.initializers.append(onnx_tensor)
    
    def _trace_graph(self, output_tensor: Tensor, input_tensors: List[Tensor], 
                     parameters: List[Tensor]):
        """追踪计算图"""
        # 标记输入和参数
        input_ids = {id(t) for t in input_tensors}
        param_ids = {id(t) for t in parameters}
        
        # 拓扑排序
        topo = []
        visited = set()
        
        def build_topo(node):
            if id(node) not in visited:
                visited.add(id(node))
                for parent in node._parents:
                    build_topo(parent)
                topo.append(node)
        
        build_topo(output_tensor)
        
        # 处理输入
        for i, inp in enumerate(input_tensors):
            name = self._get_tensor_name(inp)
            self.inputs.append(
                helper.make_tensor_value_info(
                    name,
                    self._numpy_dtype_to_onnx(inp.data.dtype),
                    list(inp.shape)
                )
            )
        
        # 处理参数（权重）
        for param in parameters:
            name = self._get_tensor_name(param)
            self._add_initializer(param, name)
        
        # 转换操作
        for tensor in topo:
            if id(tensor) in input_ids or id(tensor) in param_ids:
                continue  # 跳过输入和参数
            
            self._convert_op(tensor)
        
        # 处理输出
        output_name = self._get_tensor_name(output_tensor)
        self.outputs.append(
            helper.make_tensor_value_info(
                output_name,
                self._numpy_dtype_to_onnx(output_tensor.data.dtype),
                list(output_tensor.shape)
            )
        )
    
    def _convert_op(self, tensor: Tensor):
        """将 Nanograd 操作转换为 ONNX 节点"""
        op = tensor._op
        output_name = self._get_tensor_name(tensor)
        
        if op == '+':
            # Add
            assert len(tensor._parents) == 2
            input_names = [self._get_tensor_name(p) for p in tensor._parents]
            node = helper.make_node('Add', input_names, [output_name])
            self.nodes.append(node)
        
        elif op == '-':
            # Sub
            assert len(tensor._parents) == 2
            input_names = [self._get_tensor_name(p) for p in tensor._parents]
            node = helper.make_node('Sub', input_names, [output_name])
            self.nodes.append(node)
        
        elif op == '*':
            # Mul
            assert len(tensor._parents) == 2
            input_names = [self._get_tensor_name(p) for p in tensor._parents]
            node = helper.make_node('Mul', input_names, [output_name])
            self.nodes.append(node)
        
        elif op == '/':
            # Div
            assert len(tensor._parents) == 2
            input_names = [self._get_tensor_name(p) for p in tensor._parents]
            node = helper.make_node('Div', input_names, [output_name])
            self.nodes.append(node)
        
        elif op == '@':
            # MatMul
            assert len(tensor._parents) == 2
            input_names = [self._get_tensor_name(p) for p in tensor._parents]
            node = helper.make_node('MatMul', input_names, [output_name])
            self.nodes.append(node)
        
        elif op == 'relu':
            # ReLU
            assert len(tensor._parents) == 1
            input_name = self._get_tensor_name(tensor._parents[0])
            node = helper.make_node('Relu', [input_name], [output_name])
            self.nodes.append(node)
        
        elif op == 'sigmoid':
            # Sigmoid
            assert len(tensor._parents) == 1
            input_name = self._get_tensor_name(tensor._parents[0])
            node = helper.make_node('Sigmoid', [input_name], [output_name])
            self.nodes.append(node)
        
        elif op == 'tanh':
            # Tanh
            assert len(tensor._parents) == 1
            input_name = self._get_tensor_name(tensor._parents[0])
            node = helper.make_node('Tanh', [input_name], [output_name])
            self.nodes.append(node)
        
        elif op == 'transpose':
            # Transpose
            assert len(tensor._parents) == 1
            input_name = self._get_tensor_name(tensor._parents[0])
            # 默认转置（交换最后两个维度）
            perm = list(range(len(tensor.shape)))
            if len(perm) >= 2:
                perm[-2], perm[-1] = perm[-1], perm[-2]
            node = helper.make_node('Transpose', [input_name], [output_name], perm=perm)
            self.nodes.append(node)
        
        elif op == 'reshape':
            # Reshape
            assert len(tensor._parents) == 1
            input_name = self._get_tensor_name(tensor._parents[0])
            
            # 创建 shape 常量
            shape_name = self._get_unique_name("shape")
            shape_tensor = numpy_helper.from_array(
                np.array(tensor.shape, dtype=np.int64), 
                shape_name
            )
            self.initializers.append(shape_tensor)
            
            node = helper.make_node('Reshape', [input_name, shape_name], [output_name])
            self.nodes.append(node)
        
        elif op == 'sum':
            # ReduceSum
            assert len(tensor._parents) == 1
            input_name = self._get_tensor_name(tensor._parents[0])
            # 注意：这里简化处理，假设是全局求和
            node = helper.make_node('ReduceSum', [input_name], [output_name], keepdims=0)
            self.nodes.append(node)
        
        elif op == 'mean':
            # ReduceMean
            assert len(tensor._parents) == 1
            input_name = self._get_tensor_name(tensor._parents[0])
            node = helper.make_node('ReduceMean', [input_name], [output_name], keepdims=0)
            self.nodes.append(node)
        
        elif op == 'neg':
            # Neg
            assert len(tensor._parents) == 1
            input_name = self._get_tensor_name(tensor._parents[0])
            node = helper.make_node('Neg', [input_name], [output_name])
            self.nodes.append(node)
        
        elif op.startswith('**'):
            # Pow
            assert len(tensor._parents) == 1
            input_name = self._get_tensor_name(tensor._parents[0])
            
            # 提取指数
            exponent = float(op[2:])
            exp_name = self._get_unique_name("exponent")
            exp_tensor = numpy_helper.from_array(
                np.array([exponent], dtype=np.float32),
                exp_name
            )
            self.initializers.append(exp_tensor)
            
            node = helper.make_node('Pow', [input_name, exp_name], [output_name])
            self.nodes.append(node)
        
        else:
            # 未知操作，跳过或警告
            if op:  # 只有非空操作才警告
                print(f"警告: 不支持的操作 '{op}'，跳过")
    
    def export(self, model, input_shape: Tuple[int, ...], 
               output_path: str, model_name: str = "nanograd_model"):
        """导出模型到 ONNX
        
        Args:
            model: Nanograd 模型（需要有 forward 和 parameters 方法）
            input_shape: 输入形状（例如 (1, 784) 表示 batch_size=1, features=784）
            output_path: ONNX 文件保存路径
            model_name: 模型名称
        """
        # 创建虚拟输入进行前向传播（追踪计算图）
        dummy_input = Tensor(np.random.randn(*input_shape).astype(np.float32))
        output = model.forward(dummy_input)
        
        # 获取模型参数
        parameters = model.parameters()
        
        # 追踪计算图
        self._trace_graph(output, [dummy_input], parameters)
        
        # 创建 ONNX 图
        graph = helper.make_graph(
            self.nodes,
            model_name,
            self.inputs,
            self.outputs,
            self.initializers
        )
        
        # 创建 ONNX 模型
        onnx_model = helper.make_model(graph, producer_name="nanograd")
        onnx_model.opset_import[0].version = 13  # 使用 ONNX opset 13
        
        # 检查模型
        try:
            onnx.checker.check_model(onnx_model)
            print(f"✅ ONNX 模型验证通过")
        except Exception as e:
            print(f"⚠️  ONNX 模型验证失败: {e}")
        
        # 保存模型
        onnx.save(onnx_model, output_path)
        print(f"✅ 模型已保存到: {output_path}")
        
        # 打印模型信息
        print(f"\n模型信息:")
        print(f"  输入: {[inp.name for inp in self.inputs]}")
        print(f"  输出: {[out.name for out in self.outputs]}")
        print(f"  节点数: {len(self.nodes)}")
        print(f"  参数数: {len(self.initializers)}")
        
        return onnx_model


def export_to_onnx(model, input_shape: Tuple[int, ...], 
                   output_path: str, model_name: str = "nanograd_model"):
    """便捷函数：导出模型到 ONNX
    
    Args:
        model: Nanograd 模型（需要有 forward 和 parameters 方法）
        input_shape: 输入形状（例如 (1, 784)）
        output_path: ONNX 文件保存路径
        model_name: 模型名称
    
    Example:
        >>> model = MLP()
        >>> export_to_onnx(model, (1, 784), "model.onnx")
    """
    exporter = ONNXExporter()
    return exporter.export(model, input_shape, output_path, model_name)
