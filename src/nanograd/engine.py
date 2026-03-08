# 实现一个支持标量计算的自动微分框架
# 实现自动微分框架的三个步骤：
#           1，构建图
#           2，前向：执行图，存储计算结果
#           3，后向：反向执行图

class Value():
    """
    一个存储标量值及其梯度的节点类，用于构建动态计算图并支持自动微分。
    """
    def __init__(self, data, _children=(), _op=""):
        self.data = data  # 存储Value的值
        self.grad = 0
        self._backward = lambda: None
        self._parents = set(_children)  # 记录当前children记作为下个节点的父节点
        self._op = _op                  # 操作符 + - * / 等等

    def __add__(self, other):   # 加
        # 下面进行前向计算
        other = other if isinstance(other, Value) else Value(other)  # 确保other是value类型
        out = Value(self.data + other.data, (self, other), '+')

        def _backward(): # 定义反向执行   考虑的实际上是out的梯度是如何分配给 self 和 other  （z = x + y  又因为 ∂z/∂x = 1  ∂z/∂y = 1） 也就是说：输出 z 的梯度，会原封不动地（乘以 1）传给 x 和 y。
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward  # 使用闭包，这个节点专属的反向传播小函数”绑定到节点本身

        return out

    def __mul__(self, other):   # 乘
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():     # z = x * y 又因为 ∂z/∂x = y  ∂z/∂y = x
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"      # 先做类型检查
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if (self.data < 0) else self.data, (self,), 'Relu')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        """计算图是一个 有向无环图(DAG)，而梯度传播的方向是 逆着数据流方向
           需要使用拓扑排序， 到当前这个节点之后， 需要保证前面的节点都被执行完了"""
        topo = []
        visited = set()
        def build_topo(cur_node):
            if cur_node not in visited:
                visited.add(cur_node)
                for parent in cur_node._parents: # 遍历当前节点的父节点
                    build_topo(parent)
                topo.append(cur_node)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()


    # Python 的运算符重载（operator overloading），目的是让 Value 对象可以像普通数字一样自然地使用 + - * / 等运算符，并且支持左右两边任意一边是普通数字的情况。
    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"