from unittest import TestCase

from autograd import Value

class TestScalarOps(TestCase):
    def test_neg_grad(self):
        a = Value(2)
        c = -a
        c.backward()
        assert a.grad == -1

    def test_add_grad(self):
        a = Value(5)
        c = a + 3
        c.backward()
        assert a.grad == 1

    def test_radd_grad(self):
        a = Value(5)
        c = 3 + a
        c.backward()
        assert a.grad == 1

    def test_sub_grad(self):
        a = Value(5)
        c = a - 4
        c.backward()
        assert a.grad == 1

    def test_rsub_grad(self):
        a = Value(5)
        c = 4 - a
        c.backward()
        assert a.grad == -1

    def test_mul_grad(self):
        a = Value(5)
        c = a * 4
        c.backward()
        assert a.grad == 4

    def test_rmul_grad(self):
        a = Value(5)
        c = 4 * a
        c.backward()
        assert a.grad == 4

    def test_truediv_grad(self):
        a = Value(5)
        c = a/4
        c.backward()
        assert a.grad == (1/4)

    def test_rtruediv_grad(self):
        a = Value(5)
        c = 4/a
        c.backward()
        assert a.grad == -(4/25)

    def test_pow_grad(self):
        a = Value(5)
        c = a**2
        c.backward()
        assert a.grad == 10

    def test_chain_grad(self):
        a = Value(3)
        b = 4*a
        c = b**3 + 7
        c.backward()
        assert a.grad == 1728
        assert b.grad == 432