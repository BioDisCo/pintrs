"""Dask integration tests - ported from pint's test_dask.py.

Tests ArrayQuantity interop with dask arrays.
"""

from __future__ import annotations

import pytest
from pintrs import UnitRegistry
from pintrs.numpy_support import ArrayQuantity

np = pytest.importorskip("numpy", reason="numpy is not available")
dask = pytest.importorskip("dask", reason="dask is not available")
da = pytest.importorskip("dask.array", reason="dask.array is not available")


@pytest.fixture
def ureg() -> UnitRegistry:
    return UnitRegistry(force_ndarray_like=True)


class TestDaskArrayQuantityInterop:
    """Test that ArrayQuantity works with dask-computed numpy arrays."""

    def test_dask_computed_to_array_quantity(self, ureg: UnitRegistry) -> None:
        darr = da.arange(0, 10, chunks=5, dtype=float)
        nparr = darr.compute()
        q = ArrayQuantity(nparr, "meter", ureg)
        assert q.shape == (10,)
        assert q.magnitude[0] == pytest.approx(0.0)
        assert q.magnitude[9] == pytest.approx(9.0)

    def test_operations_on_computed_dask(self, ureg: UnitRegistry) -> None:
        darr = da.arange(0, 5, chunks=5, dtype=float)
        nparr = darr.compute()
        q = ArrayQuantity(nparr, "meter", ureg)
        result = q.to("kilometer")
        assert result.magnitude[1] == pytest.approx(0.001)

    def test_dask_compute_then_sum(self, ureg: UnitRegistry) -> None:
        darr = da.ones(10, chunks=5, dtype=float)
        nparr = darr.compute()
        q = ArrayQuantity(nparr, "kilogram", ureg)
        total = np.sum(q)
        assert total.magnitude == pytest.approx(10.0)

    def test_dask_compute_then_multiply(self, ureg: UnitRegistry) -> None:
        darr = da.arange(1, 6, chunks=5, dtype=float)
        nparr = darr.compute()
        meters = ArrayQuantity(nparr, "meter", ureg)
        seconds = ArrayQuantity(nparr, "second", ureg)
        result = meters * seconds
        assert result.units_str == "m * s"
        assert result.magnitude[0] == pytest.approx(1.0)

    def test_dask_2d_array(self, ureg: UnitRegistry) -> None:
        darr = da.arange(0, 25, chunks=5, dtype=float).reshape((5, 5))
        nparr = darr.compute()
        q = ArrayQuantity(nparr, "meter", ureg)
        assert q.shape == (5, 5)
        assert q.ndim == 2

    def test_force_ndarray_like_accepted(self) -> None:
        ureg = UnitRegistry(force_ndarray_like=True)
        assert ureg.force_ndarray_like is True

    def test_dask_compute_then_dot(self, ureg: UnitRegistry) -> None:
        a = da.arange(1, 4, chunks=3, dtype=float).compute()
        b = da.arange(4, 7, chunks=3, dtype=float).compute()
        qa = ArrayQuantity(a, "meter", ureg)
        qb = ArrayQuantity(b, "second", ureg)
        result = np.dot(qa, qb)
        expected = np.dot(a, b)
        assert result.magnitude == pytest.approx(expected)


class TestDaskLazyArrays:
    """Test dask lazy array operations that don't require pint wrapping."""

    def test_dask_add_then_convert(self, ureg: UnitRegistry) -> None:
        a = da.ones(5, chunks=5, dtype=float)
        b = da.full(5, 2.0, chunks=5)
        result = (a + b).compute()
        q = ArrayQuantity(result, "meter", ureg)
        assert np.all(q.magnitude == pytest.approx(3.0))

    def test_dask_reduction_then_quantity(self, ureg: UnitRegistry) -> None:
        darr = da.arange(10, chunks=5, dtype=float)
        mean_val = darr.mean().compute()
        q = ureg.Quantity(float(mean_val), "kelvin")
        assert q.magnitude == pytest.approx(4.5)
