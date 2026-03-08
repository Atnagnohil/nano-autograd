"""验证 PyTorch (CPU) 安装的精简脚本。"""

import sys
import torch


def verify_torch_cpu() -> bool:
    """验证 PyTorch 是否成功安装并能输出版本号与 CPU 设备信息。

    Returns:
        bool: 如果成功导入并创建基础张量，返回 True；否则返回 False。
    """
    try:
        print(f"📦 PyTorch 版本: {torch.__version__}")
        
        # 检查设备（预期为 cpu）
        device = torch.device("cpu")
        print(f"🖥️  当前测试设备: {device}")

        # 基础张量创建测试
        x = torch.tensor([1.0, 2.0, 3.0], device=device)
        print(f"✅ 成功在 {device} 上创建张量: {x}")

        print("\n🎉 PyTorch (CPU) 安装正常。")
        return True

    except ImportError:
        print("\n❌ 未找到 PyTorch，请确认已在虚拟环境中执行 `uv pip install -e .[dev]`", file=sys.stderr)
        return False
    except Exception as e:
        print(f"\n❌ 运行发生异常: {e}", file=sys.stderr)
        return False


if __name__ == "__main__":
    success = verify_torch_cpu()
    sys.exit(0 if success else 1)