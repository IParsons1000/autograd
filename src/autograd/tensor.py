from .scalar import Scalar
import numpy as np

class Tensor(Scalar):
    """ store tensor and its gradient """

    def __init__(self, data, _children=(), _op=''):
        self.data = data if isinstance(data, np.ndarray) else np.array(data, dtype=float)
        self.shape = self.data.shape
        self.ndim = self.data.ndim
        self.grad = np.zeros(self.shape, dtype=float)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.types = (int, float, np.ndarray)
        self.ones = np.ones(self.shape, dtype=float)

    """
    Utilities
    """

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

