"""Sparse array integration tests - ported from pint's test_compat_downcast.py.

Tests ArrayQuantity interop with sparse arrays.
"""

from __future__ import annotations

import pytest
from pintrs import UnitRegistry
from pintrs.numpy_support import ArrayQuantity

np = pytest.importorskip("numpy", reason="numpy is not available")
sparse = pytest.importorskip("sparse", reason="sparse is not available")


@pytest.fixture
def ureg() -> UnitRegistry:
    return UnitRegistry()


class TestSparseInterop:
    """Test ArrayQuantity with sparse arrays converted to dense."""

    def test_sparse_coo_to_dense_quantity(self, ureg: UnitRegistry) -> None:
        coords = [[0, 1, 2], [1, 0, 2]]
        data = [1.0, 2.0, 3.0]
        s = sparse.COO(coords, data, shape=(3, 3))
        dense = s.todense()
        q = ArrayQuantity(dense, "meter", ureg)
        assert q.shape == (3, 3)
        assert q.magnitude[0, 1] == pytest.approx(1.0)
        assert q.magnitude[1, 0] == pytest.approx(2.0)

    def test_sparse_operations_then_quantity(self, ureg: UnitRegistry) -> None:
        s1 = sparse.COO(np.array([[0, 1], [1, 0]]), [1.0, 2.0], shape=(3, 3))
        s2 = sparse.COO(np.array([[0, 2], [0, 2]]), [3.0, 4.0], shape=(3, 3))
        result = (s1 + s2).todense()
        q = ArrayQuantity(result, "kilogram", ureg)
        assert q.magnitude[0, 0] == pytest.approx(3.0)

    def test_sparse_multiply_dense_to_quantity(self, ureg: UnitRegistry) -> None:
        s = sparse.COO(np.eye(3))
        dense = s.todense()
        q = ArrayQuantity(dense, "meter", ureg)
        result = q * ArrayQuantity(np.array([2.0, 3.0, 4.0]), "second", ureg)
        assert result.magnitude[0, 0] == pytest.approx(2.0)
        assert result.magnitude[1, 1] == pytest.approx(3.0)

    def test_sparse_sum_to_scalar(self, ureg: UnitRegistry) -> None:
        coords = [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]]
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        s = sparse.COO(coords, data, shape=(5, 5))
        total = float(s.sum())
        q = ureg.Quantity(total, "newton")
        assert q.magnitude == pytest.approx(15.0)

    def test_sparse_dot_product(self, ureg: UnitRegistry) -> None:
        a = sparse.COO(np.array([1.0, 0.0, 3.0]))
        b = sparse.COO(np.array([2.0, 5.0, 1.0]))
        dot_val = float(sparse.tensordot(a, b, axes=1))
        q = ureg.Quantity(dot_val, "meter ** 2")
        assert q.magnitude == pytest.approx(5.0)

    def test_sparse_reshape(self, ureg: UnitRegistry) -> None:
        s = sparse.COO(np.arange(12.0).reshape(3, 4))
        dense = s.todense()
        q = ArrayQuantity(dense, "volt", ureg)
        reshaped = np.reshape(q, (12,))
        assert reshaped.shape == (12,)
        assert reshaped.magnitude[0] == pytest.approx(0.0)
        assert reshaped.magnitude[11] == pytest.approx(11.0)
