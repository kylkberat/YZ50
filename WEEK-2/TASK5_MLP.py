import math
import random
import matplotlib.pyplot as plt

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

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        bir = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        out = bir.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts # nin = 3, nouts = [4, 4, 1] ---> 3 + [4, 4, 1] = [3] + [4, 4, 1] = [3, 4, 4, 1]
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

n = MLP(3, [4, 4, 1])

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0] # İstenen hedefler

losses = []

for step in range(20):
    # forward pass
    ypred = [n(x) for x in xs]
    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

    # backward pass'ten önce gradları sıfırlamamız gerekiyor.
    for p in n.parameters():
        p.grad = 0.0

    # backward pass
    loss.backward()

    # güncelleme
    lr = 0.12
    for p in n.parameters():
        p.data -= lr * p.grad  # STEP UZUNLUĞU BURADA

    print(step, loss.data)
    losses.append(loss.data)

print(ypred)

plt.plot(losses)
plt.xlabel('adım')
plt.ylabel('loss')
plt.title('Training loss (lr=0.12)')
plt.show()
