"""Port of pint's test_umath.py for pintrs.

Tests numpy ufunc integration with ArrayQuantity, covering math operations,
trigonometric functions, comparison functions, and floating-point functions.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from pintrs import UnitRegistry
from pintrs.numpy_support import ArrayQuantity

PI = np.pi


@pytest.fixture
def ureg() -> UnitRegistry:
    """Shared unit registry for all tests."""
    return UnitRegistry()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _aq(
    values: list[float] | list[complex],
    units: str,
    ureg: UnitRegistry,
) -> ArrayQuantity:
    """Shorthand for creating an ArrayQuantity."""
    return ArrayQuantity(np.asarray(values), units, ureg)


# ---------------------------------------------------------------------------
# Math ufuncs
# ---------------------------------------------------------------------------


class TestMathUfuncs:
    """numpy math ufuncs applied to ArrayQuantity."""

    def test_add(self, ureg: UnitRegistry) -> None:
        a = _aq([1.0, 2.0, 3.0, 4.0], "joule", ureg)
        b = _aq([2.0, 4.0, 6.0, 8.0], "joule", ureg)
        result = np.add(a, b)
        np.testing.assert_allclose(result.magnitude, [3.0, 6.0, 9.0, 12.0])

    def test_subtract(self, ureg: UnitRegistry) -> None:
        a = _aq([2.0, 4.0, 6.0, 8.0], "joule", ureg)
        b = _aq([1.0, 2.0, 3.0, 4.0], "joule", ureg)
        result = np.subtract(a, b)
        np.testing.assert_allclose(result.magnitude, [1.0, 2.0, 3.0, 4.0])

    def test_multiply(self, ureg: UnitRegistry) -> None:
        a = _aq([1.0, 2.0, 3.0, 4.0], "joule", ureg)
        b = _aq([2.0, 4.0, 6.0, 8.0], "joule", ureg)
        result = np.multiply(a, b)
        np.testing.assert_allclose(result.magnitude, [2.0, 8.0, 18.0, 32.0])

    def test_multiply_scalar(self, ureg: UnitRegistry) -> None:
        a = _aq([1.0, 2.0, 3.0, 4.0], "joule", ureg)
        result = np.multiply(a, 3.0)
        np.testing.assert_allclose(result.magnitude, [3.0, 6.0, 9.0, 12.0])

    def test_divide(self, ureg: UnitRegistry) -> None:
        a = _aq([2.0, 4.0, 6.0, 8.0], "joule", ureg)
        b = _aq([1.0, 2.0, 3.0, 4.0], "joule", ureg)
        result = np.divide(a, b)
        np.testing.assert_allclose(result.magnitude, [2.0, 2.0, 2.0, 2.0])

    def test_true_divide(self, ureg: UnitRegistry) -> None:
        a = _aq([2.0, 4.0, 6.0, 8.0], "joule", ureg)
        b = _aq([1.0, 2.0, 3.0, 4.0], "joule", ureg)
        result = np.true_divide(a, b)
        np.testing.assert_allclose(result.magnitude, [2.0, 2.0, 2.0, 2.0])

    def test_floor_divide(self, ureg: UnitRegistry) -> None:
        a = _aq([5.0, 7.0, 10.0, 13.0], "joule", ureg)
        b = _aq([2.0, 3.0, 3.0, 4.0], "joule", ureg)
        result = np.floor_divide(a, b)
        np.testing.assert_allclose(result.magnitude, [2.0, 2.0, 3.0, 3.0])

    def test_negative(self, ureg: UnitRegistry) -> None:
        q = _aq([1.0, 2.0, 3.0, 4.0], "joule", ureg)
        result = np.negative(q)
        np.testing.assert_allclose(result.magnitude, [-1.0, -2.0, -3.0, -4.0])

    def test_positive(self, ureg: UnitRegistry) -> None:
        q = _aq([1.0, -2.0, 3.0, -4.0], "joule", ureg)
        result = np.positive(q)
        np.testing.assert_allclose(result.magnitude, [1.0, -2.0, 3.0, -4.0])

    def test_absolute(self, ureg: UnitRegistry) -> None:
        q = _aq([-1.0, 2.0, -3.0, 4.0], "joule", ureg)
        result = np.absolute(q)
        np.testing.assert_allclose(result.magnitude, [1.0, 2.0, 3.0, 4.0])

    def test_absolute_complex(self, ureg: UnitRegistry) -> None:
        q = _aq([1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j], "meter", ureg)
        result = np.absolute(q)
        expected = np.absolute(np.array([1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j]))
        np.testing.assert_allclose(result.magnitude, expected)

    def test_rint(self, ureg: UnitRegistry) -> None:
        q = _aq([1.4, 2.6, 3.5, 4.1], "joule", ureg)
        result = np.rint(q)
        expected = np.rint(np.array([1.4, 2.6, 3.5, 4.1]))
        np.testing.assert_allclose(result.magnitude, expected)

    def test_conj(self, ureg: UnitRegistry) -> None:
        q = _aq([1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j], "meter", ureg)
        result = np.conj(q)
        expected = np.conj(np.array([1 + 1j, 2 + 2j, 3 + 3j, 4 + 4j]))
        np.testing.assert_allclose(result.magnitude, expected)

    def test_exp(self, ureg: UnitRegistry) -> None:
        q = _aq([1.0, 2.0, 3.0, 4.0], "dimensionless", ureg)
        result = np.exp(q)
        expected = np.exp(np.array([1.0, 2.0, 3.0, 4.0]))
        np.testing.assert_allclose(result.magnitude, expected)

    def test_exp2(self, ureg: UnitRegistry) -> None:
        q = _aq([1.0, 2.0, 3.0, 4.0], "dimensionless", ureg)
        result = np.exp2(q)
        expected = np.exp2(np.array([1.0, 2.0, 3.0, 4.0]))
        np.testing.assert_allclose(result.magnitude, expected)

    def test_log(self, ureg: UnitRegistry) -> None:
        q = _aq([1.0, 2.0, 3.0, 4.0], "dimensionless", ureg)
        result = np.log(q)
        expected = np.log(np.array([1.0, 2.0, 3.0, 4.0]))
        np.testing.assert_allclose(result.magnitude, expected)

    def test_log2(self, ureg: UnitRegistry) -> None:
        q = _aq([1.0, 2.0, 4.0, 8.0], "dimensionless", ureg)
        result = np.log2(q)
        expected = np.log2(np.array([1.0, 2.0, 4.0, 8.0]))
        np.testing.assert_allclose(result.magnitude, expected)

    def test_log10(self, ureg: UnitRegistry) -> None:
        q = _aq([1.0, 10.0, 100.0, 1000.0], "dimensionless", ureg)
        result = np.log10(q)
        expected = np.log10(np.array([1.0, 10.0, 100.0, 1000.0]))
        np.testing.assert_allclose(result.magnitude, expected)

    def test_expm1(self, ureg: UnitRegistry) -> None:
        q = _aq([0.0, 0.1, 0.5, 1.0], "dimensionless", ureg)
        result = np.expm1(q)
        expected = np.expm1(np.array([0.0, 0.1, 0.5, 1.0]))
        np.testing.assert_allclose(result.magnitude, expected)

    def test_sqrt(self, ureg: UnitRegistry) -> None:
        q = _aq([1.0, 4.0, 9.0, 16.0], "m ** 2", ureg)
        result = np.sqrt(q)
        np.testing.assert_allclose(result.magnitude, [1.0, 2.0, 3.0, 4.0])

    def test_sqrt_dimensionless(self, ureg: UnitRegistry) -> None:
        q = _aq([1.0, 4.0, 9.0, 16.0], "dimensionless", ureg)
        result = np.sqrt(q)
        np.testing.assert_allclose(result.magnitude, [1.0, 2.0, 3.0, 4.0])

    def test_square(self, ureg: UnitRegistry) -> None:
        q = _aq([1.0, 2.0, 3.0, 4.0], "meter", ureg)
        result = np.square(q)
        np.testing.assert_allclose(result.magnitude, [1.0, 4.0, 9.0, 16.0])

    def test_reciprocal(self, ureg: UnitRegistry) -> None:
        q = _aq([1.0, 2.0, 4.0, 5.0], "meter", ureg)
        result = np.reciprocal(q)
        expected = np.reciprocal(np.array([1.0, 2.0, 4.0, 5.0]))
        np.testing.assert_allclose(result.magnitude, expected)

    def test_remainder(self, ureg: UnitRegistry) -> None:
        a = _aq([5.0, 7.0, 10.0, 13.0], "joule", ureg)
        b = _aq([2.0, 3.0, 3.0, 4.0], "joule", ureg)
        result = np.remainder(a, b)
        expected = np.remainder(
            np.array([5.0, 7.0, 10.0, 13.0]),
            np.array([2.0, 3.0, 3.0, 4.0]),
        )
        np.testing.assert_allclose(result.magnitude, expected)

    def test_mod(self, ureg: UnitRegistry) -> None:
        a = _aq([5.0, 7.0, 10.0, 13.0], "joule", ureg)
        b = _aq([2.0, 3.0, 3.0, 4.0], "joule", ureg)
        result = np.mod(a, b)
        expected = np.mod(
            np.array([5.0, 7.0, 10.0, 13.0]),
            np.array([2.0, 3.0, 3.0, 4.0]),
        )
        np.testing.assert_allclose(result.magnitude, expected)

    def test_fmod(self, ureg: UnitRegistry) -> None:
        a = _aq([5.0, 7.0, 10.0, 13.0], "joule", ureg)
        b = _aq([2.0, 3.0, 3.0, 4.0], "joule", ureg)
        result = np.fmod(a, b)
        expected = np.fmod(
            np.array([5.0, 7.0, 10.0, 13.0]),
            np.array([2.0, 3.0, 3.0, 4.0]),
        )
        np.testing.assert_allclose(result.magnitude, expected)


# ---------------------------------------------------------------------------
# Trigonometric ufuncs
# ---------------------------------------------------------------------------


class TestTrigUfuncs:
    """numpy trigonometric ufuncs applied to ArrayQuantity."""

    def test_sin_dimensionless(self, ureg: UnitRegistry) -> None:
        values = np.arange(0, PI / 2, PI / 4)
        q = _aq(values.tolist(), "dimensionless", ureg)
        result = np.sin(q)
        np.testing.assert_allclose(result.magnitude, np.sin(values))

    def test_sin_radian(self, ureg: UnitRegistry) -> None:
        values = np.arange(0, PI / 2, PI / 4)
        q = _aq(values.tolist(), "radian", ureg)
        result = np.sin(q)
        np.testing.assert_allclose(result.magnitude, np.sin(values))

    def test_cos_dimensionless(self, ureg: UnitRegistry) -> None:
        values = np.arange(0, PI / 2, PI / 4)
        q = _aq(values.tolist(), "dimensionless", ureg)
        result = np.cos(q)
        np.testing.assert_allclose(result.magnitude, np.cos(values))

    def test_cos_radian(self, ureg: UnitRegistry) -> None:
        values = np.arange(0, PI / 2, PI / 4)
        q = _aq(values.tolist(), "radian", ureg)
        result = np.cos(q)
        np.testing.assert_allclose(result.magnitude, np.cos(values))

    def test_tan_dimensionless(self, ureg: UnitRegistry) -> None:
        values = np.arange(0, PI / 2, PI / 4)
        q = _aq(values.tolist(), "dimensionless", ureg)
        result = np.tan(q)
        np.testing.assert_allclose(result.magnitude, np.tan(values))

    def test_tan_radian(self, ureg: UnitRegistry) -> None:
        values = np.arange(0, PI / 2, PI / 4)
        q = _aq(values.tolist(), "radian", ureg)
        result = np.tan(q)
        np.testing.assert_allclose(result.magnitude, np.tan(values))

    def test_arcsin(self, ureg: UnitRegistry) -> None:
        values = np.arange(0, 0.9, 0.1)
        q = _aq(values.tolist(), "dimensionless", ureg)
        result = np.arcsin(q)
        np.testing.assert_allclose(result.magnitude, np.arcsin(values))

    def test_arccos(self, ureg: UnitRegistry) -> None:
        values = np.arange(0, 0.9, 0.1)
        q = _aq(values.tolist(), "dimensionless", ureg)
        result = np.arccos(q)
        np.testing.assert_allclose(result.magnitude, np.arccos(values))

    def test_arctan(self, ureg: UnitRegistry) -> None:
        values = np.arange(0, 0.9, 0.1)
        q = _aq(values.tolist(), "dimensionless", ureg)
        result = np.arctan(q)
        np.testing.assert_allclose(result.magnitude, np.arctan(values))

    def test_arctan2(self, ureg: UnitRegistry) -> None:
        y_vals = np.arange(0, 0.9, 0.1)
        x_vals = np.arange(0.9, 0.0, -0.1)
        y = _aq(y_vals.tolist(), "meter", ureg)
        x = _aq(x_vals.tolist(), "meter", ureg)
        result = np.arctan2(y, x)
        np.testing.assert_allclose(
            result.magnitude,
            np.arctan2(y_vals, x_vals),
        )

    def test_hypot(self, ureg: UnitRegistry) -> None:
        a = _aq([3.0], "meter", ureg)
        b = _aq([4.0], "meter", ureg)
        result = np.hypot(a, b)
        np.testing.assert_allclose(result.magnitude, [5.0])

    def test_sinh(self, ureg: UnitRegistry) -> None:
        values = np.arange(0, PI / 2, PI / 4)
        q = _aq(values.tolist(), "dimensionless", ureg)
        result = np.sinh(q)
        np.testing.assert_allclose(result.magnitude, np.sinh(values))

    def test_sinh_radian(self, ureg: UnitRegistry) -> None:
        values = np.arange(0, PI / 2, PI / 4)
        q = _aq(values.tolist(), "radian", ureg)
        result = np.sinh(q)
        np.testing.assert_allclose(result.magnitude, np.sinh(values))

    def test_cosh(self, ureg: UnitRegistry) -> None:
        values = np.arange(0, PI / 2, PI / 4)
        q = _aq(values.tolist(), "dimensionless", ureg)
        result = np.cosh(q)
        np.testing.assert_allclose(result.magnitude, np.cosh(values))

    def test_cosh_radian(self, ureg: UnitRegistry) -> None:
        values = np.arange(0, PI / 2, PI / 4)
        q = _aq(values.tolist(), "radian", ureg)
        result = np.cosh(q)
        np.testing.assert_allclose(result.magnitude, np.cosh(values))

    def test_tanh(self, ureg: UnitRegistry) -> None:
        values = np.arange(0, PI / 2, PI / 4)
        q = _aq(values.tolist(), "dimensionless", ureg)
        result = np.tanh(q)
        np.testing.assert_allclose(result.magnitude, np.tanh(values))

    def test_tanh_radian(self, ureg: UnitRegistry) -> None:
        values = np.arange(0, PI / 2, PI / 4)
        q = _aq(values.tolist(), "radian", ureg)
        result = np.tanh(q)
        np.testing.assert_allclose(result.magnitude, np.tanh(values))

    def test_arcsinh(self, ureg: UnitRegistry) -> None:
        values = np.arange(0, 0.9, 0.1)
        q = _aq(values.tolist(), "dimensionless", ureg)
        result = np.arcsinh(q)
        np.testing.assert_allclose(result.magnitude, np.arcsinh(values))

    def test_arccosh(self, ureg: UnitRegistry) -> None:
        values = np.arange(1.0, 1.9, 0.1)
        q = _aq(values.tolist(), "dimensionless", ureg)
        result = np.arccosh(q)
        np.testing.assert_allclose(result.magnitude, np.arccosh(values))

    def test_arctanh(self, ureg: UnitRegistry) -> None:
        values = np.arange(0, 0.9, 0.1)
        q = _aq(values.tolist(), "dimensionless", ureg)
        result = np.arctanh(q)
        np.testing.assert_allclose(result.magnitude, np.arctanh(values))


# ---------------------------------------------------------------------------
# Comparison ufuncs
# ---------------------------------------------------------------------------


class TestComparisonUfuncs:
    """numpy comparison ufuncs applied to ArrayQuantity."""

    def test_greater(self, ureg: UnitRegistry) -> None:
        a = _aq([1.0, 2.0, 3.0, 4.0], "joule", ureg)
        b = _aq([2.0, 4.0, 6.0, 8.0], "joule", ureg)
        result = np.greater(a, b)
        expected = np.greater(
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([2.0, 4.0, 6.0, 8.0]),
        )
        np.testing.assert_array_equal(result, expected)

    def test_greater_equal(self, ureg: UnitRegistry) -> None:
        a = _aq([1.0, 2.0, 3.0, 4.0], "joule", ureg)
        b = _aq([1.0, 4.0, 3.0, 8.0], "joule", ureg)
        result = np.greater_equal(a, b)
        expected = np.greater_equal(
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([1.0, 4.0, 3.0, 8.0]),
        )
        np.testing.assert_array_equal(result, expected)

    def test_less(self, ureg: UnitRegistry) -> None:
        a = _aq([1.0, 2.0, 3.0, 4.0], "joule", ureg)
        b = _aq([2.0, 4.0, 6.0, 8.0], "joule", ureg)
        result = np.less(a, b)
        expected = np.less(
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([2.0, 4.0, 6.0, 8.0]),
        )
        np.testing.assert_array_equal(result, expected)

    def test_less_equal(self, ureg: UnitRegistry) -> None:
        a = _aq([1.0, 2.0, 3.0, 4.0], "joule", ureg)
        b = _aq([1.0, 4.0, 3.0, 8.0], "joule", ureg)
        result = np.less_equal(a, b)
        expected = np.less_equal(
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([1.0, 4.0, 3.0, 8.0]),
        )
        np.testing.assert_array_equal(result, expected)

    def test_not_equal(self, ureg: UnitRegistry) -> None:
        a = _aq([1.0, 2.0, 3.0, 4.0], "joule", ureg)
        b = _aq([1.0, 4.0, 3.0, 8.0], "joule", ureg)
        result = np.not_equal(a, b)
        expected = np.not_equal(
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([1.0, 4.0, 3.0, 8.0]),
        )
        np.testing.assert_array_equal(result, expected)

    def test_equal(self, ureg: UnitRegistry) -> None:
        a = _aq([1.0, 2.0, 3.0, 4.0], "joule", ureg)
        b = _aq([1.0, 4.0, 3.0, 8.0], "joule", ureg)
        result = np.equal(a, b)
        expected = np.equal(
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([1.0, 4.0, 3.0, 8.0]),
        )
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# Floating ufuncs
# ---------------------------------------------------------------------------


class TestFloatingUfuncs:
    """numpy floating-point ufuncs applied to ArrayQuantity."""

    def test_isfinite(self, ureg: UnitRegistry) -> None:
        q = _aq([1.0, 2.0, 3.0, 4.0], "joule", ureg)
        result = np.isfinite(q)
        np.testing.assert_array_equal(result, [True, True, True, True])

    def test_isinf(self, ureg: UnitRegistry) -> None:
        q = _aq([1.0, 2.0, 3.0, 4.0], "joule", ureg)
        result = np.isinf(q)
        np.testing.assert_array_equal(result, [False, False, False, False])

    def test_isnan(self, ureg: UnitRegistry) -> None:
        q = _aq([1.0, 2.0, 3.0, 4.0], "joule", ureg)
        result = np.isnan(q)
        np.testing.assert_array_equal(result, [False, False, False, False])

    def test_signbit(self, ureg: UnitRegistry) -> None:
        q = _aq([1.0, -2.0, 3.0, -4.0], "joule", ureg)
        result = np.signbit(q)
        np.testing.assert_array_equal(result, [False, True, False, True])

    def test_copysign(self, ureg: UnitRegistry) -> None:
        a = _aq([1.0, 2.0, 3.0, 4.0], "joule", ureg)
        b = _aq([-1.0, 1.0, -1.0, 1.0], "joule", ureg)
        result = np.copysign(a, b)
        np.testing.assert_allclose(result.magnitude, [-1.0, 2.0, -3.0, 4.0])

    def test_nextafter(self, ureg: UnitRegistry) -> None:
        a = _aq([1.0, 2.0, 3.0, 4.0], "joule", ureg)
        b = _aq([2.0, 4.0, 6.0, 8.0], "joule", ureg)
        result = np.nextafter(a, b)
        expected = np.nextafter(
            np.array([1.0, 2.0, 3.0, 4.0]),
            np.array([2.0, 4.0, 6.0, 8.0]),
        )
        np.testing.assert_allclose(result.magnitude, expected)

    def test_floor(self, ureg: UnitRegistry) -> None:
        q = _aq([1.4, 2.6, 3.1, 4.9], "joule", ureg)
        result = np.floor(q)
        np.testing.assert_allclose(result.magnitude, [1.0, 2.0, 3.0, 4.0])

    def test_ceil(self, ureg: UnitRegistry) -> None:
        q = _aq([1.4, 2.6, 3.1, 4.9], "joule", ureg)
        result = np.ceil(q)
        np.testing.assert_allclose(result.magnitude, [2.0, 3.0, 4.0, 5.0])

    def test_trunc(self, ureg: UnitRegistry) -> None:
        q = _aq([1.4, 2.6, -3.1, -4.9], "joule", ureg)
        result = np.trunc(q)
        np.testing.assert_allclose(result.magnitude, [1.0, 2.0, -3.0, -4.0])

    def test_floor_meter(self, ureg: UnitRegistry) -> None:
        q = _aq([1.4, 2.6, 3.1, 4.9], "meter", ureg)
        result = np.floor(q)
        np.testing.assert_allclose(result.magnitude, [1.0, 2.0, 3.0, 4.0])

    def test_ceil_meter(self, ureg: UnitRegistry) -> None:
        q = _aq([1.4, 2.6, 3.1, 4.9], "meter", ureg)
        result = np.ceil(q)
        np.testing.assert_allclose(result.magnitude, [2.0, 3.0, 4.0, 5.0])

    def test_trunc_dimensionless(self, ureg: UnitRegistry) -> None:
        q = _aq([1.4, 2.6, -3.1, -4.9], "dimensionless", ureg)
        result = np.trunc(q)
        np.testing.assert_allclose(result.magnitude, [1.0, 2.0, -3.0, -4.0])
