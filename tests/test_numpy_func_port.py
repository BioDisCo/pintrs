"""NumPy function tests - ported from pint's test_numpy_func.py.

Tests numpy function/ufunc behavior on ArrayQuantity through the public API.
Many pint-internal functions (get_op_output_unit, convert_to_consistent_units,
etc.) are not exposed by pintrs, so we test the equivalent behavior through
ArrayQuantity operations.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from pintrs import DimensionalityError, UnitRegistry
from pintrs.numpy_support import ArrayQuantity


@pytest.fixture
def ureg() -> UnitRegistry:
    return UnitRegistry()


# ---------------------------------------------------------------------------
# Ufunc-based operations (these work via __array_ufunc__)
# ---------------------------------------------------------------------------


class TestUfuncAdd:
    def test_add_same_units(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([1.0, 2.0]), "meter", ureg)
        b = ArrayQuantity(np.array([3.0, 4.0]), "meter", ureg)
        r = np.add(a, b)
        np.testing.assert_array_equal(r.magnitude, [4.0, 6.0])

    def test_add_compatible_units(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([1.0, 2.0]), "meter", ureg)
        b = ArrayQuantity(np.array([100.0, 200.0]), "centimeter", ureg)
        r = np.add(a, b)
        np.testing.assert_allclose(r.magnitude, [2.0, 4.0])

    def test_add_incompatible_raises(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([1.0, 2.0]), "meter", ureg)
        b = ArrayQuantity(np.array([1.0, 2.0]), "second", ureg)
        with pytest.raises((DimensionalityError, ValueError)):
            np.add(a, b)


class TestUfuncSubtract:
    def test_subtract_same_units(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([5.0, 6.0]), "meter", ureg)
        b = ArrayQuantity(np.array([1.0, 2.0]), "meter", ureg)
        r = np.subtract(a, b)
        np.testing.assert_array_equal(r.magnitude, [4.0, 4.0])

    def test_subtract_incompatible_raises(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([1.0]), "meter", ureg)
        b = ArrayQuantity(np.array([1.0]), "second", ureg)
        with pytest.raises((DimensionalityError, ValueError)):
            np.subtract(a, b)


class TestUfuncMultiply:
    def test_multiply_quantities(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([2.0, 3.0]), "meter", ureg)
        b = ArrayQuantity(np.array([4.0, 5.0]), "second", ureg)
        r = np.multiply(a, b)
        np.testing.assert_array_equal(r.magnitude, [8.0, 15.0])

    def test_multiply_by_scalar(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([2.0, 3.0]), "meter", ureg)
        r = np.multiply(a, 2)
        np.testing.assert_array_equal(r.magnitude, [4.0, 6.0])

    def test_multiply_by_dimensionless(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([2.0, 3.0]), "meter", ureg)
        b = ArrayQuantity(np.array([2.0, 3.0]), "dimensionless", ureg)
        r = np.multiply(a, b)
        np.testing.assert_array_equal(r.magnitude, [4.0, 9.0])


class TestUfuncDivide:
    def test_divide_same_units(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([6.0, 8.0]), "meter", ureg)
        b = ArrayQuantity(np.array([2.0, 4.0]), "meter", ureg)
        r = np.divide(a, b)
        np.testing.assert_array_equal(r.magnitude, [3.0, 2.0])

    def test_divide_different_units(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([10.0, 20.0]), "meter", ureg)
        b = ArrayQuantity(np.array([2.0, 4.0]), "second", ureg)
        r = np.divide(a, b)
        np.testing.assert_array_equal(r.magnitude, [5.0, 5.0])

    def test_true_divide(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([6.0, 8.0]), "meter", ureg)
        b = ArrayQuantity(np.array([2.0, 4.0]), "meter", ureg)
        r = np.true_divide(a, b)
        np.testing.assert_array_equal(r.magnitude, [3.0, 2.0])

    def test_floor_divide(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([7.0, 9.0]), "meter", ureg)
        b = ArrayQuantity(np.array([2.0, 4.0]), "meter", ureg)
        r = np.floor_divide(a, b)
        np.testing.assert_array_equal(r.magnitude, [3.0, 2.0])


class TestUfuncNegativePositive:
    def test_negative(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([1.0, -2.0, 3.0]), "meter", ureg)
        r = np.negative(a)
        np.testing.assert_array_equal(r.magnitude, [-1.0, 2.0, -3.0])

    def test_positive(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([1.0, -2.0, 3.0]), "meter", ureg)
        r = np.positive(a)
        np.testing.assert_array_equal(r.magnitude, [1.0, -2.0, 3.0])


class TestUfuncAbsolute:
    def test_absolute(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([-1.0, 2.0, -3.0]), "meter", ureg)
        r = np.absolute(a)
        np.testing.assert_array_equal(r.magnitude, [1.0, 2.0, 3.0])

    def test_fabs(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([-1.0, 2.0, -3.0]), "meter", ureg)
        r = np.fabs(a)
        np.testing.assert_array_equal(r.magnitude, [1.0, 2.0, 3.0])


class TestUfuncRounding:
    def test_rint(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([1.4, 2.6, 3.5]), "meter", ureg)
        r = np.rint(a)
        np.testing.assert_array_equal(r.magnitude, [1.0, 3.0, 4.0])

    def test_floor(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([1.4, 2.6, 3.5]), "meter", ureg)
        r = np.floor(a)
        np.testing.assert_array_equal(r.magnitude, [1.0, 2.0, 3.0])

    def test_ceil(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([1.4, 2.6, 3.5]), "meter", ureg)
        r = np.ceil(a)
        np.testing.assert_array_equal(r.magnitude, [2.0, 3.0, 4.0])

    def test_trunc(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([1.9, -2.9, 3.1]), "meter", ureg)
        r = np.trunc(a)
        np.testing.assert_array_equal(r.magnitude, [1.0, -2.0, 3.0])


class TestUfuncSqrtSquare:
    def test_sqrt(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([4.0, 9.0, 16.0]), "meter ** 2", ureg)
        r = np.sqrt(a)
        np.testing.assert_allclose(r.magnitude, [2.0, 3.0, 4.0])

    def test_square(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([2.0, 3.0, 4.0]), "meter", ureg)
        r = np.square(a)
        np.testing.assert_array_equal(r.magnitude, [4.0, 9.0, 16.0])

    def test_reciprocal(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([2.0, 4.0, 5.0]), "meter", ureg)
        r = np.reciprocal(a)
        np.testing.assert_allclose(r.magnitude, [0.5, 0.25, 0.2])


# ---------------------------------------------------------------------------
# Trig ufuncs (dimensionless/radian input)
# ---------------------------------------------------------------------------


class TestTrigUfuncs:
    def test_sin(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([0.0, np.pi / 2, np.pi]), "radian", ureg)
        r = np.sin(a)
        np.testing.assert_allclose(r.magnitude, [0.0, 1.0, 0.0], atol=1e-10)

    def test_cos(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([0.0, np.pi / 2, np.pi]), "radian", ureg)
        r = np.cos(a)
        np.testing.assert_allclose(r.magnitude, [1.0, 0.0, -1.0], atol=1e-10)

    def test_tan(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([0.0, np.pi / 4]), "radian", ureg)
        r = np.tan(a)
        np.testing.assert_allclose(r.magnitude, [0.0, 1.0], atol=1e-10)

    def test_sin_with_dimensionless(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([0.0, np.pi / 2]), "dimensionless", ureg)
        r = np.sin(a)
        np.testing.assert_allclose(r.magnitude, [0.0, 1.0], atol=1e-10)

    def test_sin_with_degrees_raises_or_converts(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([0.0, 90.0]), "degree", ureg)
        try:
            r = np.sin(a)
            np.testing.assert_allclose(r.magnitude, [0.0, 1.0], atol=1e-10)
        except (DimensionalityError, ValueError):
            pass

    def test_trig_with_meter_raises(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([1.0]), "meter", ureg)
        with pytest.raises((DimensionalityError, ValueError, TypeError)):
            np.sin(a)


# ---------------------------------------------------------------------------
# Exp/log ufuncs (dimensionless only)
# ---------------------------------------------------------------------------


class TestExpLogUfuncs:
    def test_exp(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([0.0, 1.0, 2.0]), "dimensionless", ureg)
        r = np.exp(a)
        np.testing.assert_allclose(r.magnitude, [1.0, np.e, np.e**2], rtol=1e-10)

    def test_exp_with_units_raises(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([1.0]), "meter", ureg)
        with pytest.raises((DimensionalityError, ValueError, TypeError)):
            np.exp(a)

    def test_log(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([1.0, np.e, np.e**2]), "dimensionless", ureg)
        r = np.log(a)
        np.testing.assert_allclose(r.magnitude, [0.0, 1.0, 2.0], atol=1e-10)

    def test_log10(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([1.0, 10.0, 100.0]), "dimensionless", ureg)
        r = np.log10(a)
        np.testing.assert_allclose(r.magnitude, [0.0, 1.0, 2.0], atol=1e-10)

    def test_log2(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([1.0, 2.0, 4.0]), "dimensionless", ureg)
        r = np.log2(a)
        np.testing.assert_allclose(r.magnitude, [0.0, 1.0, 2.0], atol=1e-10)

    def test_exp2(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([0.0, 1.0, 3.0]), "dimensionless", ureg)
        r = np.exp2(a)
        np.testing.assert_allclose(r.magnitude, [1.0, 2.0, 8.0], atol=1e-10)

    def test_expm1(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([0.0, 1.0]), "dimensionless", ureg)
        r = np.expm1(a)
        np.testing.assert_allclose(r.magnitude, [0.0, np.e - 1], atol=1e-10)


# ---------------------------------------------------------------------------
# Comparison ufuncs
# ---------------------------------------------------------------------------


class TestComparisonUfuncs:
    def test_greater(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([3.0, 1.0, 2.0]), "meter", ureg)
        b = ArrayQuantity(np.array([2.0, 2.0, 2.0]), "meter", ureg)
        r = np.greater(a, b)
        np.testing.assert_array_equal(r, [True, False, False])

    def test_greater_equal(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([3.0, 1.0, 2.0]), "meter", ureg)
        b = ArrayQuantity(np.array([2.0, 2.0, 2.0]), "meter", ureg)
        r = np.greater_equal(a, b)
        np.testing.assert_array_equal(r, [True, False, True])

    def test_less(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([1.0, 3.0, 2.0]), "meter", ureg)
        b = ArrayQuantity(np.array([2.0, 2.0, 2.0]), "meter", ureg)
        r = np.less(a, b)
        np.testing.assert_array_equal(r, [True, False, False])

    def test_less_equal(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([1.0, 3.0, 2.0]), "meter", ureg)
        b = ArrayQuantity(np.array([2.0, 2.0, 2.0]), "meter", ureg)
        r = np.less_equal(a, b)
        np.testing.assert_array_equal(r, [True, False, True])

    def test_equal(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([1.0, 2.0, 3.0]), "meter", ureg)
        b = ArrayQuantity(np.array([1.0, 3.0, 3.0]), "meter", ureg)
        r = np.equal(a, b)
        np.testing.assert_array_equal(r, [True, False, True])

    def test_not_equal(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([1.0, 2.0, 3.0]), "meter", ureg)
        b = ArrayQuantity(np.array([1.0, 3.0, 3.0]), "meter", ureg)
        r = np.not_equal(a, b)
        np.testing.assert_array_equal(r, [False, True, False])

    def test_comparison_incompatible_raises(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([1.0]), "meter", ureg)
        b = ArrayQuantity(np.array([1.0]), "second", ureg)
        with pytest.raises((DimensionalityError, ValueError)):
            np.greater(a, b)


# ---------------------------------------------------------------------------
# Floating-point inspection ufuncs
# ---------------------------------------------------------------------------


class TestFloatingUfuncs:
    def test_isfinite(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([1.0, np.inf, np.nan]), "meter", ureg)
        r = np.isfinite(a)
        np.testing.assert_array_equal(r, [True, False, False])

    def test_isinf(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([1.0, np.inf, np.nan]), "meter", ureg)
        r = np.isinf(a)
        np.testing.assert_array_equal(r, [False, True, False])

    def test_isnan(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([1.0, np.inf, np.nan]), "meter", ureg)
        r = np.isnan(a)
        np.testing.assert_array_equal(r, [False, False, True])

    def test_signbit(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([1.0, -2.0, 0.0]), "meter", ureg)
        r = np.signbit(a)
        np.testing.assert_array_equal(r, [False, True, False])

    def test_copysign(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([1.0, 2.0, 3.0]), "meter", ureg)
        b = ArrayQuantity(np.array([-1.0, 1.0, -1.0]), "meter", ureg)
        r = np.copysign(a, b)
        np.testing.assert_array_equal(r.magnitude, [-1.0, 2.0, -3.0])

    def test_nextafter(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([1.0]), "meter", ureg)
        b = ArrayQuantity(np.array([2.0]), "meter", ureg)
        r = np.nextafter(a, b)
        assert r.magnitude[0] > 1.0

    def test_isreal(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([1.0, 2.0, 3.0]), "meter", ureg)
        r = np.isreal(a)
        np.testing.assert_array_equal(r, [True, True, True])

    def test_iscomplex(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([1.0, 2.0, 3.0]), "meter", ureg)
        r = np.iscomplex(a)
        np.testing.assert_array_equal(r, [False, False, False])


# ---------------------------------------------------------------------------
# Remainder ufuncs
# ---------------------------------------------------------------------------


class TestRemainderUfuncs:
    def test_remainder(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([7.0, 8.0]), "meter", ureg)
        b = ArrayQuantity(np.array([3.0, 3.0]), "meter", ureg)
        r = np.remainder(a, b)
        np.testing.assert_array_equal(r.magnitude, [1.0, 2.0])

    def test_mod(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([7.0, 8.0]), "meter", ureg)
        b = ArrayQuantity(np.array([3.0, 3.0]), "meter", ureg)
        r = np.mod(a, b)
        np.testing.assert_array_equal(r.magnitude, [1.0, 2.0])

    def test_fmod(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([7.0, 8.0]), "meter", ureg)
        b = ArrayQuantity(np.array([3.0, 3.0]), "meter", ureg)
        r = np.fmod(a, b)
        np.testing.assert_array_equal(r.magnitude, [1.0, 2.0])


# ---------------------------------------------------------------------------
# ArrayQuantity methods (dot, prod, T, etc.)
# ---------------------------------------------------------------------------


class TestArrayQuantityMethods:
    def test_dot(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([1.0, 2.0, 3.0]), "meter", ureg)
        b = ArrayQuantity(np.array([4.0, 5.0, 6.0]), "second", ureg)
        r = a.dot(b)
        assert r == pytest.approx(32.0)

    def test_prod(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([2.0, 3.0, 4.0]), "dimensionless", ureg)
        r = a.prod()
        assert r == pytest.approx(24.0)

    def test_transpose(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([[1.0, 2.0], [3.0, 4.0]]), "meter", ureg)
        r = a.T
        np.testing.assert_array_equal(r.magnitude, [[1.0, 3.0], [2.0, 4.0]])

    def test_clip(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([1.0, 5.0, 10.0]), "meter", ureg)
        r = a.clip(2.0, 8.0)
        np.testing.assert_array_equal(r.magnitude, [2.0, 5.0, 8.0])

    def test_tolist(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([1.0, 2.0, 3.0]), "meter", ureg)
        r = a.tolist()
        assert r == [1.0, 2.0, 3.0]

    def test_shape(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([[1.0, 2.0], [3.0, 4.0]]), "meter", ureg)
        assert a.shape == (2, 2)

    def test_ndim(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([[1.0, 2.0], [3.0, 4.0]]), "meter", ureg)
        assert a.ndim == 2

    def test_dtype(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([1.0, 2.0]), "meter", ureg)
        assert a.dtype == np.float64

    def test_m_as(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([1000.0, 2000.0]), "meter", ureg)
        r = a.m_as("kilometer")
        np.testing.assert_allclose(r, [1.0, 2.0])

    def test_flat(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([[1.0, 2.0], [3.0, 4.0]]), "meter", ureg)
        f = a.flat
        assert len(f) == 4

    def test_real_imag(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([1.0 + 2.0j, 3.0 + 4.0j]), "meter", ureg)
        np.testing.assert_array_equal(a.real.magnitude, [1.0, 3.0])
        np.testing.assert_array_equal(a.imag.magnitude, [2.0, 4.0])


# ---------------------------------------------------------------------------
# Conjugate ufunc
# ---------------------------------------------------------------------------


class TestConjugate:
    def test_conj(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([1.0 + 2.0j, 3.0 - 4.0j]), "meter", ureg)
        r = np.conj(a)
        np.testing.assert_array_equal(r.magnitude, [1.0 - 2.0j, 3.0 + 4.0j])

    def test_conjugate(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([1.0 + 2.0j, 3.0 - 4.0j]), "meter", ureg)
        r = np.conjugate(a)
        np.testing.assert_array_equal(r.magnitude, [1.0 - 2.0j, 3.0 + 4.0j])


# ---------------------------------------------------------------------------
# Sign ufunc
# ---------------------------------------------------------------------------


class TestSign:
    def test_sign(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([-3.0, 0.0, 5.0]), "meter", ureg)
        r = np.sign(a)
        np.testing.assert_array_equal(r.magnitude, [-1.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# Sum and mean via ufuncs
# ---------------------------------------------------------------------------


class TestReductions:
    def test_sum(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([1.0, 2.0, 3.0]), "meter", ureg)
        r = np.sum(a)
        assert float(r.magnitude) == pytest.approx(6.0)

    def test_mean(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([2.0, 4.0, 6.0]), "meter", ureg)
        r = np.mean(a)
        assert float(r.magnitude) == pytest.approx(4.0)

    def test_min(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([3.0, 1.0, 2.0]), "meter", ureg)
        r = np.min(a)
        assert float(r.magnitude) == pytest.approx(1.0)

    def test_max(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([3.0, 1.0, 2.0]), "meter", ureg)
        r = np.max(a)
        assert float(r.magnitude) == pytest.approx(3.0)

    def test_std(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(
            np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]), "meter", ureg
        )
        r = np.std(a)
        assert float(r.magnitude) == pytest.approx(2.0)

    def test_cumsum(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([1.0, 2.0, 3.0]), "meter", ureg)
        r = np.cumsum(a)
        np.testing.assert_array_equal(r.magnitude, [1.0, 3.0, 6.0])
