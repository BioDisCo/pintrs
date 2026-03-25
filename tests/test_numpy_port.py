"""NumPy integration tests - ported from pint's test_numpy.py."""

from __future__ import annotations

import math

import pytest

np = pytest.importorskip("numpy")

from pintrs import DimensionalityError, UnitRegistry
from pintrs.numpy_support import ArrayQuantity


@pytest.fixture
def ureg() -> UnitRegistry:
    return UnitRegistry()


class TestArrayQuantityCreation:
    def test_from_list(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1, 2, 3], "meter", ureg)
        assert len(aq.magnitude) == 3
        np.testing.assert_array_equal(aq.magnitude, [1, 2, 3])

    def test_from_ndarray(self, ureg: UnitRegistry) -> None:
        arr = np.array([1.0, 2.0, 3.0])
        aq = ArrayQuantity(arr, "meter", ureg)
        np.testing.assert_array_equal(aq.magnitude, arr)

    def test_units_str(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1, 2], "meter", ureg)
        assert "meter" in str(aq.units_str) or "m" in str(aq.units_str)

    def test_shape(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity(np.zeros((2, 3)), "meter", ureg)
        assert aq.magnitude.shape == (2, 3)

    def test_dtype(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1.0, 2.0], "meter", ureg)
        assert aq.magnitude.dtype == np.float64

    def test_empty_array(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([], "meter", ureg)
        assert len(aq.magnitude) == 0


class TestArrayQuantityConversion:
    def test_to(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1000, 2000], "meter", ureg)
        result = aq.to("kilometer")
        np.testing.assert_allclose(result.magnitude, [1, 2])

    def test_to_base_units(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1, 2], "kilometer", ureg)
        result = aq.to_base_units()
        np.testing.assert_allclose(result.magnitude, [1000, 2000])

    def test_incompatible_raises(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1, 2], "meter", ureg)
        with pytest.raises(DimensionalityError):
            aq.to("second")


class TestArrayQuantityArithmetic:
    def test_add_same_units(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 2], "meter", ureg)
        b = ArrayQuantity([3, 4], "meter", ureg)
        result = a + b
        np.testing.assert_array_equal(result.magnitude, [4, 6])

    def test_add_compatible_units(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 2], "meter", ureg)
        b = ArrayQuantity([100, 200], "centimeter", ureg)
        result = a + b
        np.testing.assert_allclose(result.magnitude, [2, 4])

    def test_sub(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([5, 6], "meter", ureg)
        b = ArrayQuantity([1, 2], "meter", ureg)
        result = a - b
        np.testing.assert_array_equal(result.magnitude, [4, 4])

    def test_mul_scalar(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 2], "meter", ureg)
        result = a * 3
        np.testing.assert_array_equal(result.magnitude, [3, 6])

    def test_mul_array_quantities(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([2, 3], "meter", ureg)
        b = ArrayQuantity([4, 5], "second", ureg)
        result = a * b
        np.testing.assert_array_equal(result.magnitude, [8, 15])

    def test_div_scalar(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([6, 8], "meter", ureg)
        result = a / 2
        np.testing.assert_array_equal(result.magnitude, [3, 4])

    def test_div_array_quantities(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([10, 20], "meter", ureg)
        b = ArrayQuantity([2, 4], "second", ureg)
        result = a / b
        np.testing.assert_array_equal(result.magnitude, [5, 5])

    def test_pow(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([2, 3], "meter", ureg)
        result = a**2
        np.testing.assert_array_equal(result.magnitude, [4, 9])

    def test_neg(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, -2], "meter", ureg)
        result = -a
        np.testing.assert_array_equal(result.magnitude, [-1, 2])

    def test_abs(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([-1, 2, -3], "meter", ureg)
        result = abs(a)
        np.testing.assert_array_equal(result.magnitude, [1, 2, 3])


class TestArrayQuantityComparison:
    def test_eq(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 2], "meter", ureg)
        b = ArrayQuantity([1, 2], "meter", ureg)
        assert a == b

    def test_ne(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 2], "meter", ureg)
        b = ArrayQuantity([1, 3], "meter", ureg)
        assert a != b


class TestArrayQuantityRepresentation:
    def test_str(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1, 2, 3], "meter", ureg)
        s = str(aq)
        assert "1" in s
        assert "meter" in s or "m" in s

    def test_repr(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1, 2, 3], "meter", ureg)
        r = repr(aq)
        assert "ArrayQuantity" in r or "1" in r


class TestArrayUfunc:
    def test_add_ufunc(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 2], "meter", ureg)
        b = ArrayQuantity([3, 4], "meter", ureg)
        result = np.add(a, b)
        np.testing.assert_array_equal(result.magnitude, [4, 6])

    def test_multiply_ufunc(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([2, 3], "meter", ureg)
        result = np.multiply(a, 2)
        np.testing.assert_array_equal(result.magnitude, [4, 6])

    def test_sqrt_ufunc(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([4, 9], "meter**2", ureg)
        result = np.sqrt(a)
        np.testing.assert_allclose(result.magnitude, [2, 3])

    def test_sin_dimensionless(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([0, math.pi / 2], "radian", ureg)
        result = np.sin(a)
        np.testing.assert_allclose(result.magnitude, [0, 1], atol=1e-10)

    def test_sum(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 2, 3], "meter", ureg)
        result = np.sum(a)
        assert float(result.magnitude) == pytest.approx(6.0)

    def test_mean(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([2, 4, 6], "meter", ureg)
        result = np.mean(a)
        assert float(result.magnitude) == pytest.approx(4.0)


class TestArrayQuantityLikeCreation:
    def test_ones_like(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 2, 3], "meter", ureg)
        result = np.ones_like(a.magnitude)
        np.testing.assert_array_equal(result, [1, 1, 1])

    def test_zeros_like(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 2, 3], "meter", ureg)
        result = np.zeros_like(a.magnitude)
        np.testing.assert_array_equal(result, [0, 0, 0])


class TestArrayQuantityShapeOps:
    def test_flatten(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([[1, 2], [3, 4]]), "meter", ureg)
        flat = a.magnitude.flatten()
        np.testing.assert_array_equal(flat, [1, 2, 3, 4])

    def test_reshape(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 2, 3, 4], "meter", ureg)
        reshaped = a.magnitude.reshape(2, 2)
        assert reshaped.shape == (2, 2)

    def test_transpose(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([[1, 2], [3, 4]]), "meter", ureg)
        t = a.magnitude.T
        np.testing.assert_array_equal(t, [[1, 3], [2, 4]])


class TestArrayQuantityReduction:
    def test_min(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([3, 1, 2], "meter", ureg)
        assert np.min(a.magnitude) == 1

    def test_max(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([3, 1, 2], "meter", ureg)
        assert np.max(a.magnitude) == 3

    def test_argmin(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([3, 1, 2], "meter", ureg)
        assert np.argmin(a.magnitude) == 1

    def test_argmax(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([3, 1, 2], "meter", ureg)
        assert np.argmax(a.magnitude) == 0

    def test_cumsum(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 2, 3], "meter", ureg)
        result = np.cumsum(a.magnitude)
        np.testing.assert_array_equal(result, [1, 3, 6])

    def test_prod(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([2, 3, 4], "dimensionless", ureg)
        assert np.prod(a.magnitude) == 24

    def test_std(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([2, 4, 4, 4, 5, 5, 7, 9], "meter", ureg)
        assert np.std(a.magnitude) == pytest.approx(2.0)

    def test_var(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([2, 4, 4, 4, 5, 5, 7, 9], "meter", ureg)
        assert np.var(a.magnitude) == pytest.approx(4.0)


class TestQuantityFromList:
    def test_from_list(self, ureg: UnitRegistry) -> None:
        q1 = ureg.Quantity(1, "meter")
        q2 = ureg.Quantity(2, "meter")
        from pintrs import Quantity

        result = Quantity.from_list([q1, q2])
        assert result.magnitude[0] == pytest.approx(1.0)
        assert result.magnitude[1] == pytest.approx(2.0)

    def test_from_sequence(self, ureg: UnitRegistry) -> None:
        q1 = ureg.Quantity(1, "meter")
        q2 = ureg.Quantity(2, "meter")
        from pintrs import Quantity

        result = Quantity.from_sequence(iter([q1, q2]))
        assert result.magnitude[0] == pytest.approx(1.0)
        assert result.magnitude[1] == pytest.approx(2.0)

    def test_from_list_with_units(self, ureg: UnitRegistry) -> None:
        from pintrs import Quantity

        result = Quantity.from_list(
            [ureg.Quantity(100, "cm"), ureg.Quantity(2, "meter")], units="meter"
        )
        np.testing.assert_allclose(result.magnitude, [1.0, 2.0])
