"""训练一个简单的 MLP 解决 XOR 问题"""
import sys
sys.path.insert(0, '../src')

import numpy as np
import matplotlib.pyplot as plt
from nanograd.tensor import Tensor
from nanograd.nn import Linear, ReLU, Sigmoid
from nanograd.optim import Adam


class XORNet:
    """简单的两层神经网络"""
    def __init__(self):
        self.fc1 = Linear(2, 4)  # 输入层 -> 隐藏层
        self.relu = ReLU()
        self.fc2 = Linear(4, 1)  # 隐藏层 -> 输出层
        self.sigmoid = Sigmoid()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    
    def parameters(self):
        """返回所有可训练参数"""
        return self.fc1.parameters() + self.fc2.parameters()


def binary_cross_entropy(pred, target):
    """二元交叉熵损失"""
    # BCE = -[y*log(p) + (1-y)*log(1-p)]
    eps = 1e-7  # 防止 log(0)
    pred_data = np.clip(pred.data, eps, 1 - eps)
    loss_data = -np.mean(target * np.log(pred_data) + (1 - target) * np.log(1 - pred_data))
    
    # 创建损失张量
    loss = Tensor(loss_data, _children=(pred,), _op='bce')
    
    def _backward():
        # BCE 对 pred 的梯度: (pred - target) / (pred * (1 - pred))
        grad = (pred_data - target) / (pred_data * (1 - pred_data) + eps)
        pred.grad += grad / len(target)  # 平均
    
    loss._backward = _backward
    return loss


def train_xor():
    """训练 XOR 网络"""
    # XOR 数据集
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)
    
    # 创建模型和优化器
    model = XORNet()
    optimizer = Adam(model.parameters(), lr=0.01)
    
    # 训练
    epochs = 5000
    losses = []
    
    print("开始训练 XOR 网络...")
    for epoch in range(epochs):
        # 前向传播
        X_tensor = Tensor(X)
        pred = model.forward(X_tensor)
        loss = binary_cross_entropy(pred, y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.data)
        
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.data:.6f}")
    
    # 测试
    print("\n测试结果:")
    X_tensor = Tensor(X)
    pred = model.forward(X_tensor)
    for i in range(len(X)):
        print(f"输入: {X[i]}, 预测: {pred.data[i][0]:.4f}, 真实: {y[i][0]}")
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('XOR Training Loss')
    plt.grid(True)
    plt.savefig('xor_loss.png')
    print("\n损失曲线已保存到 xor_loss.png")
    plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    train_xor()
