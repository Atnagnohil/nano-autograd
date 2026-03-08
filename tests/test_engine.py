# 测试梯度标量是否正确
from nanograd.engine import Value

def test_basic_add():
    a = Value(2.0)
    b = Value(3.0)

    c = a + b
    c.backward()

    assert c.data == 5.0
    assert a.grad == 1.0
    assert b.grad == 1.0


def test_basic_mul():
    a = Value(2.0)
    b = Value(3.0)

    c = a * b
    c.backward()

    assert c.data == 6.0
    assert a.grad == 3.0
    assert b.grad == 2.0


def test_pow():
    a = Value(3.0)

    b = a ** 2
    b.backward()

    assert b.data == 9.0
    assert a.grad == 6.0


