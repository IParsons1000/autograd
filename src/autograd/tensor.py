import numpy as np

class Tensor:
    """ store tensor and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data if isinstance(data, np.ndarray) else np.array(data, dtype=float)
        self.shape = self.data.shape
        self.ndim = self.data.ndim
        self.grad = np.zeros(self.shape, dtype=float)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    """
    Arithmetic
    """

    # negation operator
    def __neg__(self):
        return self * -1

    # addition operator
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad = self.grad + out.grad
            other.grad = other.grad + out.grad
        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    # subtraction operator
    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    # multiplication operator
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad = self.grad + (other.data * out.grad)
            other.grad = other.grad + (self.data * out.grad)
        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    # division operator
    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    # power operator
    def __pow__(self, other):
        assert isinstance(other, (int, float, np.ndarray))
        out = Tensor(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad = self.grad + (other * self.data**(other - 1) * out.grad)
        out._backward = _backward

        return out

    """
    Utilities
    """

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    # backpropogation
    def backward(self):

        # topological order of graph nodes
        topo = []
        visited = set()

        # sort
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # apply chain rule
        self.grad = np.ones(self.shape, dtype=float)
        for v in reversed(topo):
            v._backward()
