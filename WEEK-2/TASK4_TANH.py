import math
import torch

class Value:
    def __init__(self, data, _prev=(), _op='', label=''):
        self.data = data
        self._prev = set(_prev)
        self.grad = 0.0
        self._op = _op
        self.label = label
        self._backward = lambda: None

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
        
    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * (-1)

    def __sub__(self, other):
        return self + (-other)

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "sadece int veya float girebilirsin"
        out = Value(self.data ** other, (self, ), f'**{other}')
        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
        
    def __truediv__(self, other):
        # a / b = a * (b ** -1)
        return self * (other ** -1)

    def tanh(self):
        out = Value((math.exp(self.data) - math.exp(-self.data)) / (math.exp(self.data) + math.exp(-self.data)), (self, ), 'tanh')
        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad
        out._backward = _backward
        return out

    def __repr__(self):
        return f"Value({self.data})"

    def backward(self):
        visited = set()
        nodes = []
        def recursion(node):
            if node not in visited:
                visited.add(node)
                for child_node in node._prev:
                    recursion(child_node)
                nodes.append(node)
        recursion(self)
        self.grad = 1.0
        for node in reversed(nodes):
            node._backward()


#--------------- Neuron Sonuç 1 -----------------
x1 = Value(2.0)
w1 = Value(-3.0)
x2 = Value(1.5)
w2 = Value(3.0)
b = Value(2.3813735870195432)

x1w1 = x1*w1
x2w2 = x2*w2
x1w1x2w2 = x1w1 + x2w2
n = x1w1x2w2 + b
out1 = n.tanh()  # tanh'i parçalıyoruz, bu yüzden buna out1 diyeceğim.

out1.backward()
print("out1 sonucu: ", x1.grad, w1.grad, x2.grad, w2.grad, b.grad)
#-------------------------------------------------


#----------------- Neuron Sonuç 2 ----------------
x1 = Value(2.0)
w1 = Value(-3.0)
x2 = Value(1.5)
w2 = Value(3.0)
b = Value(2.3813735870195432)

x1w1 = x1*w1
x2w2 = x2*w2
x1w1x2w2 = x1w1 + x2w2
n = x1w1x2w2 + b
# bu sefer tanh'ın (e^(2x) - 1) / (e^(2x) + 1) yazımını kullanacağız.
e = (2*n).exp()
out2 = (e - 1) / (e + 1)

out2.backward()
print("out2 sonucu: ", x1.grad, w1.grad, x2.grad, w2.grad, b.grad)
#--------------------------------------------------

#-------------------- Pytorch ---------------------
x1 = torch.tensor(2.0, requires_grad=True, dtype=torch.float64)
w1 = torch.tensor(-3.0, requires_grad=True, dtype=torch.float64)
x2 = torch.tensor(1.5, requires_grad=True, dtype=torch.float64)
w2 = torch.tensor(3.0, requires_grad=True, dtype=torch.float64)
b = torch.tensor(2.3813735870195432, requires_grad=True, dtype=torch.float64)

x1w1 = x1*w1
x2w2 = x2*w2
x1w1x2w2 = x1w1 + x2w2
n = x1w1x2w2 + b
out3 = torch.tanh(n)

out3.backward()
print("out3 sonucu: ", x1.grad.item(), w1.grad.item(), x2.grad.item(), w2.grad.item(), b.grad.item())
#--------------------------------------------------

#--------------- Numerical Derivative -------------
h = 0.00000001

def test(x1_raw, w1_raw, x2_raw, w2_raw, b_raw):
    x1 = Value(x1_raw); w1 = Value(w1_raw); x2 = Value(x2_raw); w2 = Value(w2_raw)
    b = Value(b_raw)

    x1w1 = x1*w1
    x2w2 = x2*w2
    x1w1x2w2 = x1w1 + x2w2
    n = x1w1x2w2 + b
    res1 = n.tanh()
    return res1.data

x1_grad = ((test(2.0 + h, -3.0, 1.5, 3.0, 2.3813735870195432)) - test(2.0, -3.0, 1.5, 3.0, 2.3813735870195432)) / h
w1_grad = ((test(2.0, -3.0 + h, 1.5, 3.0, 2.3813735870195432)) - test(2.0, -3.0, 1.5, 3.0, 2.3813735870195432)) / h
x2_grad = ((test(2.0, -3.0, 1.5 + h, 3.0, 2.3813735870195432)) - test(2.0, -3.0, 1.5, 3.0, 2.3813735870195432)) / h
w2_grad = ((test(2.0, -3.0, 1.5, 3.0 + h, 2.3813735870195432)) - test(2.0, -3.0, 1.5, 3.0, 2.3813735870195432)) / h
b_grad = ((test(2.0, -3.0, 1.5, 3.0, 2.3813735870195432 + h)) - test(2.0, -3.0, 1.5, 3.0, 2.3813735870195432)) / h
print("out4 sonucu: ", x1_grad, w1_grad, x2_grad, w2_grad, b_grad)
#--------------------------------------------------