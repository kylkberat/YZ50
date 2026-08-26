import math
from graphviz import Digraph

class Value:
    def __init__(self, data, _prev=(), _op='', label=''):
        self.data = data
        self._prev = set(_prev)
        self.grad = 0.0
        self._op = _op
        self.label = label

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return Value(self.data + other.data, (self, other), '+')

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return Value(self.data * other.data, (self, other), '*')

    def __repr__(self):
        return f"Value({self.data})"

    def tanh(self):
        return Value((math.exp(self.data) - math.exp(-self.data)) / (math.exp(self.data) + math.exp(-self.data)), (self, ), 'tanh')


#---------------- Değer Atamaları ---------------
a = Value(3.0)
b = Value(4.2)
c = Value(2.5)
d = Value(5.0)
k = Value(2.4)
g = Value(4.0)

e = a * b
f = c + d
j = k + e
i = f * g
L = j + i

a.label = 'a'; b.label = 'b'; c.label = 'c'; d.label = 'd'; e.label = 'e';
f.label = 'f'; g.label = 'g'; k.label = 'k'; i.label = 'i'; j.label = 'j';
L.label = 'L'

"""
dL/dj = 1; dL/di = 1;

dj/dk = 1 * 1 = 1; dj/de = 1 * 1 = 1;
de/da = b * 1 = 4.2; de/db = a * 1 = 3.0;
-----
di/dg = f * 1 = 7.5; di/df = g * 1 = 4.0;
df/dc = 4 * 1 = 4; df/dd = 4 * 1 = 4;
"""

L.grad = 1.0
j.grad = 1.0; i.grad = 1.0
k.grad = 1.0; e.grad = 1.0; f.grad = 4.0; g.grad = 7.5
a.grad = 4.2; b.grad = 3.0; c.grad = 4.0; d.grad = 4.0
#------------------------------------------------


#--------------- Test Fonksiyonu ----------------
def test(a_raw, b_raw, c_raw, d_raw, k_raw, g_raw):
    a = Value(a_raw); b = Value(b_raw); c = Value(c_raw); d = Value(d_raw)
    k = Value(k_raw); g = Value(g_raw)

    e = a * b
    f = c + d
    j = k + e
    i = f * g
    L = j + i

    return L.data
#------------------------------------------------


#-------------- Numerical Derivative ------------
h = 0.0001
a_grad = (test(3.0 + h, 4.2, 2.5, 5.0, 2.4, 4.0) - test(3.0, 4.2, 2.5, 5.0, 2.4, 4.0)) / h
b_grad = (test(3.0, 4.2 + h, 2.5, 5.0, 2.4, 4.0) - test(3.0, 4.2, 2.5, 5.0, 2.4, 4.0)) / h
c_grad = (test(3.0, 4.2, 2.5 + h, 5.0, 2.4, 4.0) - test(3.0, 4.2, 2.5, 5.0, 2.4, 4.0)) / h
d_grad = (test(3.0, 4.2, 2.5, 5.0 + h, 2.4, 4.0) - test(3.0, 4.2, 2.5, 5.0, 2.4, 4.0)) / h
k_grad = (test(3.0, 4.2, 2.5, 5.0, 2.4 + h, 4.0) - test(3.0, 4.2, 2.5, 5.0, 2.4, 4.0)) / h
g_grad = (test(3.0, 4.2, 2.5, 5.0, 2.4, 4.0 + h) - test(3.0, 4.2, 2.5, 5.0, 2.4, 4.0)) / h

print(a_grad, b_grad, c_grad, d_grad, k_grad, g_grad)
#-------------------------------------------------


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
sum = x1w1x2w2 + b
out = sum.tanh()

out.grad = 1.0
sum.grad = 1 - out.data ** 2  # Bu değer 0.5'e eşit.
x1w1.grad = sum.grad * 1
x2w2.grad = sum.grad * 1
b.grad = sum.grad * 1
x1.grad = x1w1.grad * w1.data
w1.grad = x1w1.grad * x1.data
x2.grad = x2w2.grad * w2.data
w2.grad = x2w2.grad * x2.data

print(x1.grad, w1.grad, x2.grad, w2.grad, x1w1.grad, x2w2.grad, sum.grad, out.grad, b.grad)
#-------------------------------------------------


#--------------- Gezinme Fonksiyonu --------------
def gezin(root):
    visited = set()
    edges = set()

    def recursion(node):
        if node not in visited:
            visited.add(node)
            for child_node in node._prev:
                edges.add((child_node, node))
                recursion(child_node)
    recursion(root)
    return visited, edges

print(gezin(L))
#-------------------------------------------------


#------------------ Graphviz ---------------------
# Graphviz Kodu Claude Opus'tan alındı. Gezin
# fonksiyonu tamamen bana ait.
def ciz(root, gezin_fonksiyonu):
    dugumler, kenarlar = gezin_fonksiyonu(root)
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})

    for n in dugumler:
        uid = str(id(n))
        dot.node(name=uid,
                 label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad),
                 shape='record')
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)

    for cocuk, ebeveyn in kenarlar:
        dot.edge(str(id(cocuk)), str(id(ebeveyn)) + ebeveyn._op)

    return dot
g = ciz(L, gezin)
g.render('graf', view=True)
#-------------------------------------------------