class Value:
    def __init__(self, data, _prev=(), _op=''):
        self.data = data
        self._prev = set(_prev)
        self.grad = 0.0
        self._op = _op

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


a = Value(3.0)
b = Value(4.5)
c = a + b
d = c * b

print(c + 1.0)
print(d * 12)

e = a + a
print(e)
print(len(e._prev))