"""Tests for numpy array quantity support."""

from __future__ import annotations

import pytest

numpy = pytest.importorskip("numpy")

from pintrs import UnitRegistry
from pintrs.numpy_support import ArrayQuantity


@pytest.fixture
def ureg() -> UnitRegistry:
    return UnitRegistry()


class TestArrayCreation:
    def test_from_array(self, ureg: UnitRegistry) -> None:
        q = ArrayQuantity(numpy.array([1.0, 2.0, 3.0]), "meter", ureg)
        assert q.shape == (3,)
        assert q.ndim == 1
        numpy.testing.assert_array_equal(q.magnitude, [1.0, 2.0, 3.0])

    def test_from_list(self, ureg: UnitRegistry) -> None:
        q = ArrayQuantity([1.0, 2.0], "meter", ureg)
        assert q.shape == (2,)

    def test_2d(self, ureg: UnitRegistry) -> None:
        q = ArrayQuantity(numpy.array([[1, 2], [3, 4]]), "meter", ureg)
        assert q.shape == (2, 2)
        assert q.ndim == 2


class TestArrayConversion:
    def test_to(self, ureg: UnitRegistry) -> None:
        q = ArrayQuantity(numpy.array([1.0, 2.0]), "kilometer", ureg)
        result = q.to("meter")
        numpy.testing.assert_array_almost_equal(result.magnitude, [1000.0, 2000.0])

    def test_m_as(self, ureg: UnitRegistry) -> None:
        q = ArrayQuantity(numpy.array([1.0, 2.0]), "kilometer", ureg)
        result = q.m_as("meter")
        numpy.testing.assert_array_almost_equal(result, [1000.0, 2000.0])

    def test_to_base_units(self, ureg: UnitRegistry) -> None:
        q = ArrayQuantity(numpy.array([1.0, 2.0]), "kilometer", ureg)
        result = q.to_base_units()
        numpy.testing.assert_array_almost_equal(result.magnitude, [1000.0, 2000.0])

    def test_to_compact(self, ureg: UnitRegistry) -> None:
        q = ArrayQuantity(numpy.array([1500, 2500]), "meter", ureg)
        result = q.to_compact()
        numpy.testing.assert_array_almost_equal(result.magnitude, [1.5, 2.5])


class TestArrayArithmetic:
    def test_add_same_units(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(numpy.array([1.0, 2.0]), "meter", ureg)
        b = ArrayQuantity(numpy.array([3.0, 4.0]), "meter", ureg)
        result = a + b
        numpy.testing.assert_array_equal(result.magnitude, [4.0, 6.0])

    def test_add_compatible_units(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(numpy.array([1.0, 2.0]), "kilometer", ureg)
        b = ArrayQuantity(numpy.array([500.0, 1000.0]), "meter", ureg)
        result = a + b
        numpy.testing.assert_array_almost_equal(result.magnitude, [1.5, 3.0])

    def test_sub(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(numpy.array([10.0, 20.0]), "meter", ureg)
        b = ArrayQuantity(numpy.array([3.0, 5.0]), "meter", ureg)
        result = a - b
        numpy.testing.assert_array_equal(result.magnitude, [7.0, 15.0])

    def test_mul_scalar(self, ureg: UnitRegistry) -> None:
        q = ArrayQuantity(numpy.array([1.0, 2.0]), "meter", ureg)
        result = q * 3
        numpy.testing.assert_array_equal(result.magnitude, [3.0, 6.0])

    def test_mul_quantity(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(numpy.array([2.0, 3.0]), "meter", ureg)
        b = ArrayQuantity(numpy.array([4.0, 5.0]), "second", ureg)
        result = a * b
        numpy.testing.assert_array_equal(result.magnitude, [8.0, 15.0])

    def test_div(self, ureg: UnitRegistry) -> None:
        q = ArrayQuantity(numpy.array([10.0, 20.0]), "meter", ureg)
        result = q / 2
        numpy.testing.assert_array_equal(result.magnitude, [5.0, 10.0])

    def test_pow(self, ureg: UnitRegistry) -> None:
        q = ArrayQuantity(numpy.array([2.0, 3.0]), "meter", ureg)
        result = q**2
        numpy.testing.assert_array_equal(result.magnitude, [4.0, 9.0])

    def test_neg(self, ureg: UnitRegistry) -> None:
        q = ArrayQuantity(numpy.array([1.0, -2.0]), "meter", ureg)
        result = -q
        numpy.testing.assert_array_equal(result.magnitude, [-1.0, 2.0])

    def test_abs(self, ureg: UnitRegistry) -> None:
        q = ArrayQuantity(numpy.array([-1.0, 2.0]), "meter", ureg)
        result = abs(q)
        numpy.testing.assert_array_equal(result.magnitude, [1.0, 2.0])


class TestArrayComparison:
    def test_gt(self, ureg: UnitRegistry) -> None:
        q = ArrayQuantity(numpy.array([1.0, 2.0, 3.0]), "meter", ureg)
        result = q > ArrayQuantity(numpy.array([1.5, 1.5, 1.5]), "meter", ureg)
        numpy.testing.assert_array_equal(result, [False, True, True])

    def test_eq(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(numpy.array([1.0, 2.0]), "meter", ureg)
        b = ArrayQuantity(numpy.array([1.0, 2.0]), "meter", ureg)
        numpy.testing.assert_array_equal(a == b, [True, True])


class TestArrayIndexing:
    def test_getitem_scalar(self, ureg: UnitRegistry) -> None:
        q = ArrayQuantity(numpy.array([1.0, 2.0, 3.0]), "meter", ureg)
        assert q[0].magnitude == 1.0

    def test_getitem_slice(self, ureg: UnitRegistry) -> None:
        q = ArrayQuantity(numpy.array([1.0, 2.0, 3.0]), "meter", ureg)
        result = q[1:]
        assert isinstance(result, ArrayQuantity)
        numpy.testing.assert_array_equal(result.magnitude, [2.0, 3.0])


class TestArrayProperties:
    def test_dimensionality(self, ureg: UnitRegistry) -> None:
        q = ArrayQuantity(numpy.array([1.0]), "meter", ureg)
        assert q.dimensionality == "[length]"

    def test_dimensionless(self, ureg: UnitRegistry) -> None:
        q = ArrayQuantity(numpy.array([1.0]), "dimensionless", ureg)
        assert q.dimensionless
        assert q.unitless

    def test_transpose(self, ureg: UnitRegistry) -> None:
        q = ArrayQuantity(numpy.array([[1, 2], [3, 4]]), "meter", ureg)
        result = q.T
        assert result.shape == (2, 2)
        assert result.magnitude[0][1] == 3

    def test_dtype(self, ureg: UnitRegistry) -> None:
        q = ArrayQuantity(numpy.array([1.0, 2.0]), "meter", ureg)
        assert q.dtype == numpy.float64


class TestArrayNumpy:
    def test_sqrt(self, ureg: UnitRegistry) -> None:
        q = ArrayQuantity(numpy.array([4.0, 9.0]), "m ** 2", ureg)
        result = numpy.sqrt(q)
        numpy.testing.assert_array_almost_equal(result.magnitude, [2.0, 3.0])

    def test_multiply(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(numpy.array([2.0, 3.0]), "meter", ureg)
        result = numpy.multiply(a, 2)
        numpy.testing.assert_array_equal(result.magnitude, [4.0, 6.0])

    def test_array_protocol(self, ureg: UnitRegistry) -> None:
        q = ArrayQuantity(numpy.array([1.0, 2.0]), "meter", ureg)
        arr = numpy.asarray(q)
        numpy.testing.assert_array_equal(arr, [1.0, 2.0])
