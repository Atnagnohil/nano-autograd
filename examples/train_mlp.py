"""训练多层感知机 (MLP) 在 MNIST 数据集上"""
import sys
sys.path.insert(0, '../src')

import numpy as np
import matplotlib.pyplot as plt
from nanograd.tensor import Tensor
from nanograd.nn import Linear, ReLU
from nanograd.optim import Adam, SGD


class MLP:
    """多层感知机"""
    def __init__(self, input_size=784, hidden_sizes=[128, 64], num_classes=10):
        self.layers = []
        
        # 构建网络层
        sizes = [input_size] + hidden_sizes + [num_classes]
        for i in range(len(sizes) - 1):
            self.layers.append(Linear(sizes[i], sizes[i+1]))
            if i < len(sizes) - 2:  # 最后一层不加激活
                self.layers.append(ReLU())
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        """返回所有可训练参数"""
        params = []
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                params.extend(layer.parameters())
        return params


def softmax(x):
    """Softmax 函数（用于预测）"""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy_loss(logits, targets):
    """交叉熵损失
    
    Args:
        logits: 模型输出 (batch_size, num_classes)
        targets: 目标标签 (batch_size,) 整数标签
    """
    batch_size = len(targets)
    
    # Softmax
    probs = softmax(logits.data)
    
    # 交叉熵
    log_probs = np.log(probs + 1e-8)
    loss_data = -np.mean([log_probs[i, int(targets[i])] for i in range(batch_size)])
    
    # 创建损失张量
    loss = Tensor(loss_data, _children=(logits,), _op='cross_entropy')
    
    def _backward():
        # Softmax + CrossEntropy 的梯度: (softmax - one_hot) / batch_size
        grad = probs.copy()
        for i in range(batch_size):
            grad[i, int(targets[i])] -= 1
        logits.grad += grad / batch_size
    
    loss._backward = _backward
    return loss


def accuracy(logits, targets):
    """计算准确率"""
    preds = np.argmax(logits, axis=1)
    return np.mean(preds == targets)


def load_mnist_simple():
    """加载 MNIST 数据集（简化版）
    
    如果没有数据文件，生成一个小的模拟数据集用于测试
    """
    try:
        from utils.Read_MNIST_Tool import load_train_images, load_train_labels, load_test_images, load_test_labels
        
        print("正在加载 MNIST 数据集...")
        train_images = load_train_images()
        train_labels = load_train_labels()
        test_images = load_test_images()
        test_labels = load_test_labels()
        
        # 归一化到 [0, 1]
        train_images = train_images.reshape(-1, 784) / 255.0
        test_images = test_images.reshape(-1, 784) / 255.0
        
        print(f"训练集: {train_images.shape}, 测试集: {test_images.shape}")
        return train_images, train_labels, test_images, test_labels
    
    except Exception as e:
        print(f"无法加载 MNIST 数据: {e}")
        print("使用模拟数据集...")
        
        # 生成模拟数据
        np.random.seed(42)
        train_images = np.random.randn(1000, 784).astype(np.float32) * 0.1
        train_labels = np.random.randint(0, 10, 1000)
        test_images = np.random.randn(200, 784).astype(np.float32) * 0.1
        test_labels = np.random.randint(0, 10, 200)
        
        return train_images, train_labels, test_images, test_labels


def train_mnist(epochs=10, batch_size=32, lr=0.001):
    """训练 MNIST 分类器"""
    # 加载数据
    train_images, train_labels, test_images, test_labels = load_mnist_simple()
    
    # 创建模型
    model = MLP(input_size=784, hidden_sizes=[128, 64], num_classes=10)
    optimizer = Adam(model.parameters(), lr=lr)
    
    # 训练历史
    train_losses = []
    train_accs = []
    test_accs = []
    
    num_batches = len(train_images) // batch_size
    
    print(f"\n开始训练 MLP (epochs={epochs}, batch_size={batch_size}, lr={lr})...")
    print("=" * 60)
    
    for epoch in range(epochs):
        # 打乱数据
        indices = np.random.permutation(len(train_images))
        train_images_shuffled = train_images[indices]
        train_labels_shuffled = train_labels[indices]
        
        epoch_loss = 0
        epoch_acc = 0
        
        # 批次训练
        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = start + batch_size
            
            batch_x = train_images_shuffled[start:end]
            batch_y = train_labels_shuffled[start:end]
            
            # 前向传播
            x_tensor = Tensor(batch_x)
            logits = model.forward(x_tensor)
            loss = cross_entropy_loss(logits, batch_y)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 记录
            epoch_loss += loss.data
            epoch_acc += accuracy(logits.data, batch_y)
        
        # 平均
        epoch_loss /= num_batches
        epoch_acc /= num_batches
        
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # 测试集评估
        test_x = Tensor(test_images)
        test_logits = model.forward(test_x)
        test_acc = accuracy(test_logits.data, test_labels)
        test_accs.append(test_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Loss: {epoch_loss:.4f} | "
              f"Train Acc: {epoch_acc:.4f} | "
              f"Test Acc: {test_acc:.4f}")
    
    print("=" * 60)
    print(f"训练完成！最终测试准确率: {test_accs[-1]:.4f}")
    
    # 绘制训练曲线
    plot_training_curves(train_losses, train_accs, test_accs)
    
    return model, train_losses, train_accs, test_accs


def plot_training_curves(losses, train_accs, test_accs):
    """绘制训练曲线"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 损失曲线
    ax1.plot(losses, label='Training Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    ax2.plot(train_accs, label='Training Accuracy', linewidth=2)
    ax2.plot(test_accs, label='Test Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mnist_training.png', dpi=150)
    print("\n训练曲线已保存到 mnist_training.png")
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    train_mnist(epochs=10, batch_size=32, lr=0.001)
