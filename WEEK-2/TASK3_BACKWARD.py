import math

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

    def __repr__(self):
        return f"Value({self.data})"

    def tanh(self):
        out = Value((math.exp(self.data) - math.exp(-self.data)) / (math.exp(self.data) + math.exp(-self.data)), (self, ), 'tanh')
        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad
        out._backward = _backward
        return out

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


#------------------ Neuron -----------------------
# Değerler Claude Opus'tan alındı.
x1 = Value(2.0)
w1 = Value(-3.0)
x2 = Value(1.5)
w2 = Value(3.0)
b = Value(2.3813735870195432)

x1w1 = x1*w1
x2w2 = x2*w2
x1w1x2w2 = x1w1 + x2w2
n = x1w1x2w2 + b
out = n.tanh()


a = Value(2.5)
b = Value(3.0)
c = Value(2.0)

e = a + a
d = b * c
k = e * d
L = k * d

# print(x1.grad, w1.grad, x2.grad, w2.grad, x1w1.grad, x2w2.grad, n.grad, out.grad, b.grad)
#-------------------------------------------------

L.backward()

print(a.grad, b.grad, c.grad, d.grad, e.grad, k.grad, L.grad)
    
