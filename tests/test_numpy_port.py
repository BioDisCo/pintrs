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
        # == is elementwise like pint/numpy.
        a = ArrayQuantity([1, 2], "meter", ureg)
        b = ArrayQuantity([1, 2], "meter", ureg)
        np.testing.assert_array_equal(a == b, [True, True])

    def test_ne(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 2], "meter", ureg)
        b = ArrayQuantity([1, 3], "meter", ureg)
        np.testing.assert_array_equal(a != b, [False, True])


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


class TestArrayQuantityIndexingSlicing:
    """Tests for __getitem__, __setitem__, slicing, and fancy indexing."""

    def test_getitem_scalar(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([10, 20, 30], "meter", ureg)
        result = aq[1]
        assert float(result.magnitude) == pytest.approx(20.0)

    def test_getitem_slice(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([10, 20, 30, 40], "meter", ureg)
        result = aq[1:3]
        assert isinstance(result, ArrayQuantity)
        np.testing.assert_array_equal(result.magnitude, [20, 30])

    def test_getitem_2d_row(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity(np.array([[1, 2], [3, 4]]), "meter", ureg)
        row = aq[0]
        assert isinstance(row, ArrayQuantity)
        np.testing.assert_array_equal(row.magnitude, [1, 2])

    def test_getitem_2d_element(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity(np.array([[1, 2], [3, 4]]), "meter", ureg)
        elem = aq.magnitude[1, 1]
        assert elem == 4

    def test_getitem_boolean_mask(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1, 2, 3, 4], "meter", ureg)
        mask = np.array([True, False, True, False])
        result = aq[mask]
        assert isinstance(result, ArrayQuantity)
        np.testing.assert_array_equal(result.magnitude, [1, 3])

    def test_getitem_fancy_index(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([10, 20, 30, 40], "meter", ureg)
        result = aq[np.array([0, 3])]
        assert isinstance(result, ArrayQuantity)
        np.testing.assert_array_equal(result.magnitude, [10, 40])

    def test_setitem_scalar(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1.0, 2.0, 3.0], "meter", ureg)
        aq[0] = 99.0
        assert aq.magnitude[0] == pytest.approx(99.0)

    def test_setitem_slice(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1.0, 2.0, 3.0, 4.0], "meter", ureg)
        aq[1:3] = 0.0
        np.testing.assert_array_equal(aq.magnitude, [1.0, 0.0, 0.0, 4.0])

    def test_setitem_with_quantity(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1.0, 2.0, 3.0], "meter", ureg)
        aq[0] = ureg.Quantity(99.0, "meter")
        assert aq.magnitude[0] == pytest.approx(99.0)

    def test_setitem_with_array_quantity(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1.0, 2.0, 3.0, 4.0], "meter", ureg)
        aq[1:3] = ArrayQuantity([10.0, 20.0], "meter", ureg)
        np.testing.assert_array_equal(aq.magnitude, [1.0, 10.0, 20.0, 4.0])

    def test_setitem_ellipsis(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1.0, 2.0, 3.0], "meter", ureg)
        aq[...] = 7.0
        np.testing.assert_array_equal(aq.magnitude, [7.0, 7.0, 7.0])


class TestArrayQuantityIteration:
    """Tests for __iter__ and __len__."""

    def test_iter_1d(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1, 2, 3, 4], "meter", ureg)
        values = list(aq)
        assert len(values) == 4
        for i, v in enumerate(values):
            assert float(v.magnitude) == pytest.approx(float(i + 1))

    def test_iter_2d(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity(np.array([[1, 2], [3, 4]]), "meter", ureg)
        rows = list(aq)
        assert len(rows) == 2
        assert isinstance(rows[0], ArrayQuantity)
        np.testing.assert_array_equal(rows[0].magnitude, [1, 2])
        np.testing.assert_array_equal(rows[1].magnitude, [3, 4])

    def test_len(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1, 2, 3], "meter", ureg)
        assert len(aq) == 3

    def test_len_2d(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity(np.zeros((3, 4)), "meter", ureg)
        assert len(aq) == 3


class TestArrayQuantityComparisonOps:
    """Tests for comparison operators (lt, le, gt, ge) with units."""

    def test_lt_same_units(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 2, 3, 4], "meter", ureg)
        result = a < ArrayQuantity([2, 2, 4, 4], "meter", ureg)
        np.testing.assert_array_equal(result, [True, False, True, False])

    def test_gt_same_units(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 2, 3, 4], "meter", ureg)
        result = a > ArrayQuantity([0, 2, 2, 5], "meter", ureg)
        np.testing.assert_array_equal(result, [True, False, True, False])

    def test_le(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 2, 3], "meter", ureg)
        result = a <= ArrayQuantity([1, 3, 2], "meter", ureg)
        np.testing.assert_array_equal(result, [True, True, False])

    def test_ge(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 2, 3], "meter", ureg)
        result = a >= ArrayQuantity([1, 3, 2], "meter", ureg)
        np.testing.assert_array_equal(result, [True, False, True])

    def test_lt_compatible_units(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 2], "meter", ureg)
        b = ArrayQuantity([50, 300], "centimeter", ureg)
        result = a < b
        np.testing.assert_array_equal(result, [False, True])

    def test_gt_compatible_units(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 2], "meter", ureg)
        b = ArrayQuantity([50, 300], "centimeter", ureg)
        result = a > b
        np.testing.assert_array_equal(result, [True, False])

    def test_comparison_with_scalar_dimensionless(self, ureg: UnitRegistry) -> None:
        # Dimensionless arrays compare against a bare number by base-unit value.
        a = ArrayQuantity([1, 2, 3, 4], "", ureg)
        np.testing.assert_array_equal(a > 2, [False, False, True, True])
        # Scaled-dimensionless array: must compare in base units (issue #5).
        scaled = ArrayQuantity([4.0], "cm/ms", ureg) / ArrayQuantity(
            [1.0], "m/s", ureg
        )  # == 40
        np.testing.assert_array_equal(scaled < 41, [True])
        np.testing.assert_array_equal(scaled < 5, [False])

    def test_comparison_dimensional_with_scalar_raises(
        self, ureg: UnitRegistry
    ) -> None:
        # Comparing a dimensional array to a non-zero bare number is an error
        # (pint parity), but comparison against zero is allowed (sign check).
        a = ArrayQuantity([1, 2, 3, 4], "meter", ureg)
        with pytest.raises(DimensionalityError):
            _ = a > 2
        np.testing.assert_array_equal(a > 0, [True, True, True, True])
        np.testing.assert_array_equal(
            ArrayQuantity([-1, 1], "meter", ureg) < 0, [True, False]
        )


class TestArrayUfuncPower:
    """Tests for power, square, cbrt ufuncs."""

    def test_power_ufunc(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([2, 3], "meter", ureg)
        result = np.power(a, 2)
        np.testing.assert_array_equal(result.magnitude, [4, 9])

    def test_power_ufunc_units(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([2, 3], "meter", ureg)
        result = np.power(a, 2)
        expected_units = str(ureg.Unit("meter") ** 2)
        assert str(result.units) == expected_units

    def test_square_ufunc(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([2, 3], "meter", ureg)
        result = np.square(a)
        np.testing.assert_array_equal(result.magnitude, [4, 9])

    def test_cbrt_ufunc(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([8, 27], "meter**3", ureg)
        result = np.cbrt(a)
        np.testing.assert_allclose(result.magnitude, [2, 3])


class TestArrayUfuncExpLog:
    """Tests for exp, log ufuncs requiring dimensionless input."""

    def test_exp_dimensionless(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([0, 1, 2], "dimensionless", ureg)
        result = np.exp(a)
        np.testing.assert_allclose(result.magnitude, [1.0, math.e, math.e**2])

    def test_exp_with_units_raises(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 2], "meter", ureg)
        with pytest.raises(DimensionalityError):
            np.exp(a)

    def test_log_dimensionless(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, math.e, math.e**2], "dimensionless", ureg)
        result = np.log(a)
        np.testing.assert_allclose(result.magnitude, [0, 1, 2])

    def test_log_with_units_raises(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 2], "meter", ureg)
        with pytest.raises(DimensionalityError):
            np.log(a)

    def test_log10_dimensionless(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 10, 100], "dimensionless", ureg)
        result = np.log10(a)
        np.testing.assert_allclose(result.magnitude, [0, 1, 2])

    def test_log2_dimensionless(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 2, 4], "dimensionless", ureg)
        result = np.log2(a)
        np.testing.assert_allclose(result.magnitude, [0, 1, 2])

    def test_exp2_dimensionless(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([0, 1, 3], "dimensionless", ureg)
        result = np.exp2(a)
        np.testing.assert_allclose(result.magnitude, [1, 2, 8])

    def test_exp2_with_units_raises(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 2], "meter", ureg)
        with pytest.raises(DimensionalityError):
            np.exp2(a)


class TestArrayUfuncTrig:
    """Tests for trigonometric ufuncs with unit handling."""

    def test_cos_radians(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([0, math.pi], "radian", ureg)
        result = np.cos(a)
        np.testing.assert_allclose(result.magnitude, [1, -1], atol=1e-10)

    def test_tan_radians(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([0, math.pi / 4], "radian", ureg)
        result = np.tan(a)
        np.testing.assert_allclose(result.magnitude, [0, 1], atol=1e-10)

    def test_sin_with_units_raises(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 2], "meter", ureg)
        with pytest.raises(DimensionalityError):
            np.sin(a)

    def test_arcsin_dimensionless(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([0, 1], "dimensionless", ureg)
        result = np.arcsin(a)
        np.testing.assert_allclose(result.magnitude, [0, math.pi / 2], atol=1e-10)

    def test_arccos_dimensionless(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 0], "dimensionless", ureg)
        result = np.arccos(a)
        np.testing.assert_allclose(result.magnitude, [0, math.pi / 2], atol=1e-10)

    def test_arctan_dimensionless(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([0, 1], "dimensionless", ureg)
        result = np.arctan(a)
        np.testing.assert_allclose(result.magnitude, [0, math.pi / 4], atol=1e-10)

    def test_arcsin_with_units_raises(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([0.5], "meter", ureg)
        with pytest.raises(DimensionalityError):
            np.arcsin(a)


class TestArrayUfuncComparison:
    """Tests for comparison ufuncs (np.greater, np.less, etc.)."""

    def test_greater(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 2, 3], "meter", ureg)
        b = ArrayQuantity([2, 2, 2], "meter", ureg)
        result = np.greater(a, b)
        np.testing.assert_array_equal(result, [False, False, True])

    def test_less(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 2, 3], "meter", ureg)
        b = ArrayQuantity([2, 2, 2], "meter", ureg)
        result = np.less(a, b)
        np.testing.assert_array_equal(result, [True, False, False])

    def test_equal_ufunc(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 2, 3], "meter", ureg)
        b = ArrayQuantity([1, 5, 3], "meter", ureg)
        result = np.equal(a, b)
        np.testing.assert_array_equal(result, [True, False, True])

    def test_not_equal_ufunc(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 2, 3], "meter", ureg)
        b = ArrayQuantity([1, 5, 3], "meter", ureg)
        result = np.not_equal(a, b)
        np.testing.assert_array_equal(result, [False, True, False])

    def test_greater_equal(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 2, 3], "meter", ureg)
        b = ArrayQuantity([2, 2, 2], "meter", ureg)
        result = np.greater_equal(a, b)
        np.testing.assert_array_equal(result, [False, True, True])

    def test_less_equal(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 2, 3], "meter", ureg)
        b = ArrayQuantity([2, 2, 2], "meter", ureg)
        result = np.less_equal(a, b)
        np.testing.assert_array_equal(result, [True, True, False])

    def test_maximum_ufunc(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 4], "meter", ureg)
        b = ArrayQuantity([3, 2], "meter", ureg)
        result = np.maximum(a, b)
        np.testing.assert_array_equal(result.magnitude, [3, 4])

    def test_minimum_ufunc(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 4], "meter", ureg)
        b = ArrayQuantity([3, 2], "meter", ureg)
        result = np.minimum(a, b)
        np.testing.assert_array_equal(result.magnitude, [1, 2])


class TestArrayQuantityBroadcasting:
    """Tests for broadcasting behavior with quantities."""

    def test_add_broadcast_scalar(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 2, 3], "meter", ureg)
        b = ureg.Quantity(10, "meter")
        result = a + b
        np.testing.assert_array_equal(result.magnitude, [11, 12, 13])

    def test_mul_broadcast_scalar(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 2, 3], "meter", ureg)
        result = a * 5
        np.testing.assert_array_equal(result.magnitude, [5, 10, 15])

    def test_rmul_scalar(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 2, 3], "meter", ureg)
        result = 5 * a
        np.testing.assert_array_equal(result.magnitude, [5, 10, 15])

    def test_sub_broadcast_scalar_quantity(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([10, 20, 30], "meter", ureg)
        b = ureg.Quantity(5, "meter")
        result = a - b
        np.testing.assert_array_equal(result.magnitude, [5, 15, 25])

    def test_div_broadcast_scalar(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([10, 20, 30], "meter", ureg)
        result = a / 10
        np.testing.assert_allclose(result.magnitude, [1, 2, 3])

    def test_rdiv_scalar(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 2, 4], "meter", ureg)
        result = 8 / a
        np.testing.assert_allclose(result.magnitude, [8, 4, 2])


class TestArrayQuantityNanHandling:
    """Tests for NaN handling in array quantities."""

    def test_nan_in_array(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1, np.nan, 3], "meter", ureg)
        assert np.isnan(aq.magnitude[1])

    def test_isnan_ufunc(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1, np.nan, 3], "meter", ureg)
        result = np.isnan(aq)
        np.testing.assert_array_equal(result, [False, True, False])

    def test_isinf_ufunc(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1, np.inf, -np.inf], "meter", ureg)
        result = np.isinf(aq)
        np.testing.assert_array_equal(result, [False, True, True])

    def test_isfinite_ufunc(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1, np.nan, np.inf], "meter", ureg)
        result = np.isfinite(aq)
        np.testing.assert_array_equal(result, [True, False, False])

    def test_nan_sum(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1, np.nan, 3], "meter", ureg)
        result = np.nansum(aq.magnitude)
        assert result == pytest.approx(4.0)

    def test_nan_mean(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1, np.nan, 3], "meter", ureg)
        result = np.nanmean(aq.magnitude)
        assert result == pytest.approx(2.0)

    def test_nan_min(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1, np.nan, 3], "meter", ureg)
        assert np.nanmin(aq.magnitude) == pytest.approx(1.0)

    def test_nan_max(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1, np.nan, 3], "meter", ureg)
        assert np.nanmax(aq.magnitude) == pytest.approx(3.0)


class TestArrayQuantityConcatenation:
    """Tests for np.concatenate and np.stack with array quantities."""

    def test_concatenate_same_units(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 2], "meter", ureg)
        b = ArrayQuantity([3, 4], "meter", ureg)
        result = np.concatenate([a, b])
        assert isinstance(result, ArrayQuantity)
        np.testing.assert_array_equal(result.magnitude, [1, 2, 3, 4])

    def test_concatenate_2d(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([[1, 2]]), "meter", ureg)
        b = ArrayQuantity(np.array([[3, 4]]), "meter", ureg)
        result = np.concatenate([a, b], axis=0)
        np.testing.assert_array_equal(result.magnitude, [[1, 2], [3, 4]])

    def test_stack_same_units(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 2], "meter", ureg)
        b = ArrayQuantity([3, 4], "meter", ureg)
        result = np.stack([a, b])
        assert isinstance(result, ArrayQuantity)
        np.testing.assert_array_equal(result.magnitude, [[1, 2], [3, 4]])


class TestArrayQuantityWhere:
    """Tests for np.where with array quantities."""

    def test_where_basic(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 2, 3, 4], "meter", ureg)
        b = ArrayQuantity([10, 20, 30, 40], "meter", ureg)
        cond = np.array([True, False, True, False])
        result = ArrayQuantity(np.where(cond, a.magnitude, b.magnitude), "meter", ureg)
        assert isinstance(result, ArrayQuantity)
        np.testing.assert_array_equal(result.magnitude, [1, 20, 3, 40])

    def test_where_with_scalar(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 2, 3, 4], "meter", ureg)
        cond = np.array([True, False, True, False])
        result = ArrayQuantity(np.where(cond, a.magnitude, 0), "meter", ureg)
        np.testing.assert_array_equal(result.magnitude, [1, 0, 3, 0])

    def test_where_from_comparison(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 2, 3, 4], "meter", ureg)
        b = ArrayQuantity([0, 0, 0, 0], "meter", ureg)
        cond = a.magnitude >= 3
        result = ArrayQuantity(np.where(cond, a.magnitude, b.magnitude), "meter", ureg)
        np.testing.assert_array_equal(result.magnitude, [0, 0, 3, 4])

    def test_where_magnitude_roundtrip(self, ureg: UnitRegistry) -> None:
        """Test np.where via magnitude extraction and re-wrapping."""
        a = ArrayQuantity([10, 20, 30, 40], "meter", ureg)
        b = ArrayQuantity([1, 2, 3, 4], "meter", ureg)
        cond = np.array([True, True, False, False])
        result = ArrayQuantity(np.where(cond, a.magnitude, b.magnitude), "meter", ureg)
        np.testing.assert_array_equal(result.magnitude, [10, 20, 3, 4])


class TestArrayQuantityClip:
    """Tests for clip operations."""

    def test_clip_min(self, ureg: UnitRegistry) -> None:
        # bare bounds require a dimensionless array (pint parity).
        aq = ArrayQuantity([1, 2, 3, 4, 5], "", ureg)
        result = aq.clip(min=3)
        np.testing.assert_array_equal(result.magnitude, [3, 3, 3, 4, 5])

    def test_clip_max(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1, 2, 3, 4, 5], "", ureg)
        result = aq.clip(max=3)
        np.testing.assert_array_equal(result.magnitude, [1, 2, 3, 3, 3])

    def test_clip_both(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1, 2, 3, 4, 5], "", ureg)
        result = aq.clip(min=2, max=4)
        np.testing.assert_array_equal(result.magnitude, [2, 2, 3, 4, 4])

    def test_np_clip(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1, 2, 3, 4, 5], "", ureg)
        result = np.clip(aq, 2, 4)
        assert isinstance(result, ArrayQuantity)
        np.testing.assert_array_equal(result.magnitude, [2, 2, 3, 4, 4])

    def test_clip_dimensional_bare_bounds_raises(self, ureg: UnitRegistry) -> None:
        # Dimensional array with bare bounds raises, like pint.
        aq = ArrayQuantity([1, 2, 3], "meter", ureg)
        with pytest.raises(DimensionalityError):
            np.clip(aq, 2, 4)
        with pytest.raises(DimensionalityError):
            _ = aq.clip(2, 4)


class TestArrayQuantitySorting:
    """Tests for sort, argsort operations."""

    def test_np_sort(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([4, 1, 3, 2], "meter", ureg)
        result = np.sort(aq)
        assert isinstance(result, ArrayQuantity)
        np.testing.assert_array_equal(result.magnitude, [1, 2, 3, 4])

    def test_np_argsort(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([4, 1, 3, 2], "meter", ureg)
        result = np.argsort(aq)
        np.testing.assert_array_equal(result, [1, 3, 2, 0])

    def test_np_sort_2d(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity(np.array([[3, 1], [4, 2]]), "meter", ureg)
        result = np.sort(aq, axis=1)
        np.testing.assert_array_equal(result.magnitude, [[1, 3], [2, 4]])


class TestArrayQuantityFillPutSearchsorted:
    """Tests for fill, put, searchsorted methods."""

    def test_fill_scalar(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1.0, 2.0, 3.0], "meter", ureg)
        aq.fill(7.0)
        np.testing.assert_array_equal(aq.magnitude, [7.0, 7.0, 7.0])

    def test_fill_with_quantity(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1.0, 2.0, 3.0], "meter", ureg)
        aq.fill(ureg.Quantity(5.0, "meter"))
        np.testing.assert_array_equal(aq.magnitude, [5.0, 5.0, 5.0])

    def test_put_scalar(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1.0, 2.0, 3.0, 4.0], "meter", ureg)
        aq.put(np.array([0, 2]), 99.0)
        np.testing.assert_array_equal(aq.magnitude, [99.0, 2.0, 99.0, 4.0])

    def test_put_with_quantity(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1.0, 2.0, 3.0, 4.0], "meter", ureg)
        aq.put(
            np.array([0, 2]),
            ArrayQuantity([10.0, 20.0], "meter", ureg),
        )
        np.testing.assert_array_equal(aq.magnitude, [10.0, 2.0, 20.0, 4.0])

    def test_searchsorted_basic(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1.0, 2.0, 3.0, 4.0], "meter", ureg)
        result = aq.searchsorted(2.5)
        assert result == 2

    def test_searchsorted_with_quantity(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1.0, 2.0, 3.0, 4.0], "meter", ureg)
        result = aq.searchsorted(ureg.Quantity(2.5, "meter"))
        assert result == 2

    def test_searchsorted_array(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1.0, 2.0, 3.0, 4.0], "meter", ureg)
        result = aq.searchsorted(
            ArrayQuantity([1.5, 3.5], "meter", ureg),
        )
        np.testing.assert_array_equal(result, [1, 3])


class TestArrayQuantityComplex:
    """Tests for complex number handling."""

    def test_complex_creation(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1 + 2j, 3 + 4j], "meter", ureg)
        assert aq.magnitude.dtype == np.complex128

    def test_real_property(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1 + 2j, 3 + 4j], "meter", ureg)
        result = aq.real
        np.testing.assert_array_equal(result.magnitude, [1, 3])

    def test_imag_property(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1 + 2j, 3 + 4j], "meter", ureg)
        result = aq.imag
        np.testing.assert_array_equal(result.magnitude, [2, 4])

    def test_conjugate(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1 + 2j, 3 + 4j], "meter", ureg)
        result = ArrayQuantity(np.conjugate(aq.magnitude), "meter", ureg)
        np.testing.assert_array_equal(result.magnitude, [1 - 2j, 3 - 4j])

    def test_isreal(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1 + 0j, 3 + 4j], "meter", ureg)
        result = np.isreal(aq)
        np.testing.assert_array_equal(result, [True, False])

    def test_iscomplex(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1 + 0j, 3 + 4j], "meter", ureg)
        result = np.iscomplex(aq)
        np.testing.assert_array_equal(result, [False, True])


class TestArrayQuantityToCompact:
    """Tests for to_compact on array quantities."""

    def test_to_compact_large(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([5000, 10000], "meter", ureg)
        result = aq.to_compact()
        assert isinstance(result, ArrayQuantity)
        assert len(result.magnitude) == 2
        # The magnitude should be rescaled to a compact unit
        assert result.magnitude[0] < 5000 or result.magnitude[1] < 10000

    def test_to_compact_small(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([0.001, 0.002], "meter", ureg)
        result = aq.to_compact()
        assert isinstance(result, ArrayQuantity)
        assert len(result.magnitude) == 2


class TestArrayQuantityCompatibility:
    """Tests for is_compatible_with on array quantities."""

    def test_compatible_same_dimension(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1, 2], "meter", ureg)
        assert aq.is_compatible_with("kilometer")

    def test_compatible_string(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1, 2], "meter", ureg)
        assert aq.is_compatible_with("centimeter")

    def test_incompatible(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1, 2], "meter", ureg)
        assert not aq.is_compatible_with("second")

    def test_compatible_with_array_quantity(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1, 2], "meter", ureg)
        other = ArrayQuantity([3, 4], "kilometer", ureg)
        assert aq.is_compatible_with(other)

    def test_incompatible_with_array_quantity(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1, 2], "meter", ureg)
        other = ArrayQuantity([3, 4], "second", ureg)
        assert not aq.is_compatible_with(other)


class TestArrayQuantityProperties:
    """Tests for various properties (shape, ndim, dtype, T, flat)."""

    def test_ndim(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity(np.zeros((2, 3)), "meter", ureg)
        assert aq.ndim == 2

    def test_ndim_1d(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1, 2, 3], "meter", ureg)
        assert aq.ndim == 1

    def test_T_property(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity(np.array([[1, 2], [3, 4]]), "meter", ureg)
        result = aq.T
        assert isinstance(result, ArrayQuantity)
        np.testing.assert_array_equal(result.magnitude, [[1, 3], [2, 4]])

    def test_flat_property(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity(np.array([[1, 2], [3, 4]]), "meter", ureg)
        flat_vals = list(aq.flat)
        assert flat_vals == [1, 2, 3, 4]

    def test_dimensionality(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1, 2], "meter", ureg)
        assert aq.dimensionality is not None

    def test_dimensionless_true(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1, 2], "dimensionless", ureg)
        assert aq.dimensionless

    def test_dimensionless_false(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1, 2], "meter", ureg)
        assert not aq.dimensionless


class TestArrayQuantityInPlaceConversion:
    """Tests for in-place conversion (ito)."""

    def test_ito(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1000, 2000], "meter", ureg)
        aq.ito("kilometer")
        np.testing.assert_allclose(aq.magnitude, [1, 2])
        assert "kilometer" in aq.units_str or "km" in aq.units_str

    def test_m_as(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1, 2], "kilometer", ureg)
        result = aq.m_as("meter")
        np.testing.assert_allclose(result, [1000, 2000])


class TestArrayQuantityMiscUfuncs:
    """Tests for misc ufuncs: abs, negative, floor_divide, remainder."""

    def test_abs_ufunc(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([-1, 2, -3], "meter", ureg)
        result = np.abs(a)
        np.testing.assert_array_equal(result.magnitude, [1, 2, 3])

    def test_negative_ufunc(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, -2, 3], "meter", ureg)
        result = np.negative(a)
        np.testing.assert_array_equal(result.magnitude, [-1, 2, -3])

    def test_floor_divide(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([7, 8, 9], "meter", ureg)
        b = ArrayQuantity([2, 3, 4], "second", ureg)
        result = np.floor_divide(a, b)
        np.testing.assert_array_equal(result.magnitude, [3, 2, 2])

    def test_remainder(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([7, 8, 9], "meter", ureg)
        b = ArrayQuantity([3, 3, 4], "meter", ureg)
        result = np.remainder(a, b)
        np.testing.assert_array_equal(result.magnitude, [1, 2, 1])

    def test_reciprocal(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1.0, 2.0, 4.0], "meter", ureg)
        result = np.reciprocal(a)
        np.testing.assert_allclose(result.magnitude, [1.0, 0.5, 0.25])

    def test_multiply_ufunc_two_quantities(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([2, 3], "meter", ureg)
        b = ArrayQuantity([4, 5], "second", ureg)
        result = np.multiply(a, b)
        np.testing.assert_array_equal(result.magnitude, [8, 15])

    def test_divide_ufunc(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([10, 20], "meter", ureg)
        b = ArrayQuantity([2, 5], "second", ureg)
        result = np.divide(a, b)
        np.testing.assert_array_equal(result.magnitude, [5, 4])

    def test_signbit_ufunc(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([-1.0, 0.0, 1.0], "meter", ureg)
        result = np.signbit(a)
        np.testing.assert_array_equal(result, [True, False, False])

    def test_hypot_ufunc(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([3.0], "meter", ureg)
        b = ArrayQuantity([4.0], "meter", ureg)
        result = np.hypot(a, b)
        np.testing.assert_allclose(result.magnitude, [5.0])


class TestArrayQuantityReshapeOps:
    """Tests for np.reshape, np.sort, np.concatenate via __array_function__."""

    def test_np_reshape(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1, 2, 3, 4], "meter", ureg)
        result = np.reshape(aq, (2, 2))
        assert isinstance(result, ArrayQuantity)
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result.magnitude, [[1, 2], [3, 4]])

    def test_ones_like_via_function(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1, 2, 3], "meter", ureg)
        result = np.ones_like(aq)
        assert isinstance(result, ArrayQuantity)
        np.testing.assert_array_equal(result.magnitude, [1, 1, 1])

    def test_zeros_like_via_function(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1, 2, 3], "meter", ureg)
        result = np.zeros_like(aq)
        assert isinstance(result, ArrayQuantity)
        np.testing.assert_array_equal(result.magnitude, [0, 0, 0])

    def test_full_like_via_function(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1, 2, 3], "meter", ureg)
        result = np.full_like(aq, 7)
        assert isinstance(result, ArrayQuantity)
        np.testing.assert_array_equal(result.magnitude, [7, 7, 7])


class TestArrayQuantityAddSubtractUnitChecking:
    """Tests for add/subtract with incompatible units raising errors."""

    def test_add_incompatible_raises(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 2], "meter", ureg)
        b = ArrayQuantity([1, 2], "second", ureg)
        with pytest.raises(DimensionalityError):
            np.add(a, b)

    def test_subtract_incompatible_raises(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 2], "meter", ureg)
        b = ArrayQuantity([1, 2], "second", ureg)
        with pytest.raises(DimensionalityError):
            np.subtract(a, b)

    def test_add_compatible_converts(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([1, 2], "meter", ureg)
        b = ArrayQuantity([100, 200], "centimeter", ureg)
        result = np.add(a, b)
        np.testing.assert_allclose(result.magnitude, [2, 4])

    def test_subtract_compatible_converts(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([2, 4], "meter", ureg)
        b = ArrayQuantity([100, 200], "centimeter", ureg)
        result = np.subtract(a, b)
        np.testing.assert_allclose(result.magnitude, [1, 2])

    def test_add_number_to_scaled_dimensionless(self, ureg: UnitRegistry) -> None:
        # Issue #5 (ArrayQuantity path): adding a bare number to a dimensionless
        # array that carries intermediate scaled units must convert to base units.
        a = ArrayQuantity([4.0], "cm/ms", ureg) / ArrayQuantity([1.0], "m/s", ureg)
        result = (a + 1).to_base_units()
        np.testing.assert_allclose(result.magnitude, [41.0])
        assert result.dimensionless
        result_r = (1 - a).to_base_units()
        np.testing.assert_allclose(result_r.magnitude, [-39.0])

    def test_add_number_to_dimensional_raises(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity([5.0], "meter", ureg)
        with pytest.raises(DimensionalityError):
            _ = a + 1
        with pytest.raises(DimensionalityError):
            _ = a - 1

    def test_add_zero_allowed_for_any_dimension(self, ureg: UnitRegistry) -> None:
        # Adding zero is allowed for any quantity and keeps its units (pint).
        a = ArrayQuantity([1.0, 2.0], "meter", ureg)
        np.testing.assert_allclose((a + 0).magnitude, [1.0, 2.0])
        assert str((a + 0).units) == "meter"
        np.testing.assert_allclose((a + np.zeros(2)).magnitude, [1.0, 2.0])
        np.testing.assert_allclose((0 - a).magnitude, [-1.0, -2.0])
        # Non-zero array operand on a dimensional quantity still raises.
        with pytest.raises(DimensionalityError):
            _ = a + np.array([0.0, 5.0])


class TestArrayQuantityMedianPercentile:
    """Tests for np.median and np.percentile."""

    def test_median(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1, 3, 2, 5, 4], "meter", ureg)
        result = np.median(aq)
        assert float(result.magnitude) == pytest.approx(3.0)

    def test_percentile(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1, 2, 3, 4], "meter", ureg)
        result = np.percentile(aq, 50)
        assert float(result.magnitude) == pytest.approx(2.5)

    def test_percentile_25(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1, 2, 3, 4], "meter", ureg)
        result = np.percentile(aq, 25)
        assert float(result.magnitude) == pytest.approx(1.75)


class TestArrayQuantityDtypeConversion:
    """Tests for dtype and tolist."""

    def test_int_dtype(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity(np.array([1, 2, 3], dtype=np.int32), "meter", ureg)
        assert aq.dtype == np.int32

    def test_float32_dtype(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity(np.array([1, 2, 3], dtype=np.float32), "meter", ureg)
        assert aq.dtype == np.float32

    def test_tolist(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1, 2, 3], "meter", ureg)
        result = aq.tolist()
        assert result == [1, 2, 3]

    def test_array_protocol(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity([1.0, 2.0, 3.0], "meter", ureg)
        arr = np.asarray(aq)
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])


class TestArrayQuantityReductionAxes:
    """Tests for reduction operations with axis parameter."""

    def test_sum_axis0(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity(np.array([[1, 2], [3, 4]]), "meter", ureg)
        result = np.sum(aq.magnitude, axis=0)
        np.testing.assert_array_equal(result, [4, 6])

    def test_sum_axis1(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity(np.array([[1, 2], [3, 4]]), "meter", ureg)
        result = np.sum(aq.magnitude, axis=1)
        np.testing.assert_array_equal(result, [3, 7])

    def test_mean_axis(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity(np.array([[1, 2], [3, 4]]), "meter", ureg)
        result = np.mean(aq.magnitude, axis=0)
        np.testing.assert_allclose(result, [2, 3])

    def test_std_axis(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity(np.array([[1, 2], [3, 4]]), "meter", ureg)
        result = np.std(aq.magnitude, axis=0)
        np.testing.assert_allclose(result, [1, 1])

    def test_var_axis(self, ureg: UnitRegistry) -> None:
        aq = ArrayQuantity(np.array([[1, 2], [3, 4]]), "meter", ureg)
        result = np.var(aq.magnitude, axis=0)
        np.testing.assert_allclose(result, [1, 1])


class TestPublicArrayBareArithmetic:
    """Issue #5: the public ``ureg.Quantity(ndarray, ...)`` array path must
    convert dimensionless-scaled values to base units when combined with a
    bare number, and support %, //, **, divmod and ordered comparisons."""

    @staticmethod
    def _scaled(ureg: UnitRegistry) -> object:
        # [40, 50] dimensionless, stored with intermediate units cm*s/m/ms.
        return ureg.Quantity(np.array([4.0, 5.0]), "cm/ms") / ureg.Quantity(1.0, "m/s")

    def test_add_sub_bare(self, ureg: UnitRegistry) -> None:
        a = self._scaled(ureg)
        np.testing.assert_allclose((a + 2).to_base_units().magnitude, [42.0, 52.0])
        np.testing.assert_allclose((2 - a).to_base_units().magnitude, [-38.0, -48.0])
        np.testing.assert_allclose(
            (a + np.array([1.0, 1.0])).to_base_units().magnitude, [41.0, 51.0]
        )

    def test_floordiv_mod_bare(self, ureg: UnitRegistry) -> None:
        a = self._scaled(ureg)
        np.testing.assert_allclose((a // 2).to_base_units().magnitude, [20.0, 25.0])
        np.testing.assert_allclose((a % 2).to_base_units().magnitude, [0.0, 0.0])
        np.testing.assert_allclose((2 // a).to_base_units().magnitude, [0.0, 0.0])

    def test_pow_bare(self, ureg: UnitRegistry) -> None:
        a = self._scaled(ureg)
        np.testing.assert_allclose((a**2).to_base_units().magnitude, [1600.0, 2500.0])

    def test_divmod_bare(self, ureg: UnitRegistry) -> None:
        a = self._scaled(ureg)
        d, m = divmod(a, 2)
        np.testing.assert_allclose(d.to_base_units().magnitude, [20.0, 25.0])
        np.testing.assert_allclose(m.to_base_units().magnitude, [0.0, 0.0])

    def test_comparison_bare(self, ureg: UnitRegistry) -> None:
        a = self._scaled(ureg)
        np.testing.assert_array_equal(a < 41, [True, False])
        np.testing.assert_array_equal(a > 45, [False, True])

    def test_dimensional_bare_raises(self, ureg: UnitRegistry) -> None:
        m = ureg.Quantity(np.array([1.0, 2.0]), "meter")
        with pytest.raises(DimensionalityError):
            _ = m + 2
        with pytest.raises(DimensionalityError):
            _ = m % 2
        with pytest.raises(DimensionalityError):
            _ = m < 2
        # Zero is allowed for any dimension.
        np.testing.assert_allclose((m + 0).magnitude, [1.0, 2.0])
        np.testing.assert_array_equal(m > 0, [True, True])


class TestArrayDimensionlessClipMatmul:
    """Issue #5 clip parity + matmul support."""

    def test_clip_scaled_dimensionless_bare_bounds(self, ureg: UnitRegistry) -> None:
        # Bare clip bounds on a dimensionless-scaled array are interpreted as
        # dimensionless values (operate in base units), matching pint.
        # base values are [0.5, 2, 5].
        a = ArrayQuantity(np.array([0.05, 0.2, 0.5]), "cm*s/m/ms", ureg)
        result = np.clip(a, 1, 3).to_base_units()
        np.testing.assert_allclose(result.magnitude, [1.0, 2.0, 3.0])

    def test_matmul(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([[1.0, 2.0]]), "meter", ureg)
        b = ArrayQuantity(np.array([[100.0], [200.0]]), "centimeter", ureg)
        result = a @ b
        np.testing.assert_allclose(result.magnitude, [[500.0]])
        # units are the product (meter * centimeter)
        units_str = str(result.units)
        assert "meter" in units_str
        assert "centimeter" in units_str


class TestScaledDimensionlessNumpyFuncs:
    """Issue #5: numpy functions that mix a bare operand with a scaled-
    dimensionless quantity must operate on base-unit values (pint parity)."""

    @staticmethod
    def _scaled(ureg: UnitRegistry) -> ArrayQuantity:
        return ArrayQuantity(
            np.array([0.05, 0.2, 0.5]), "cm*s/m/ms", ureg
        )  # base [0.5,2,5]

    def test_fmax_fmin_bare(self, ureg: UnitRegistry) -> None:
        a = self._scaled(ureg)  # base [0.5, 2, 5]
        np.testing.assert_allclose(
            np.fmax(a, 2).to_base_units().magnitude, [2.0, 2.0, 5.0]
        )
        np.testing.assert_allclose(
            np.fmin(a, 2).to_base_units().magnitude, [0.5, 2.0, 2.0]
        )

    def test_where_bare_branch(self, ureg: UnitRegistry) -> None:
        a = self._scaled(ureg)
        result = np.where([True, False, True], a, 3).to_base_units()
        np.testing.assert_allclose(result.magnitude, [0.5, 3.0, 5.0])

    def test_isclose_allclose_bare(self, ureg: UnitRegistry) -> None:
        a = self._scaled(ureg)
        np.testing.assert_array_equal(
            np.isclose(a, [0.5, 2.0, 5.0]), [True, True, True]
        )
        assert np.allclose(a, [0.5, 2.0, 5.0])

    def test_comparison_ufunc_bare_list(self, ureg: UnitRegistry) -> None:
        a = self._scaled(ureg)
        np.testing.assert_array_equal(
            np.greater(a, [0.4, 2.1, 4.9]), [True, False, True]
        )

    def test_interp_matches_numpy(self, ureg: UnitRegistry) -> None:
        a = self._scaled(ureg)  # base fp = [0.5, 2, 5]
        result = np.interp([-1, 0.5, 3], [0, 1, 2], a, left=1, right=3).to_base_units()
        # Matches raw numpy on base-unit values.
        np.testing.assert_allclose(
            result.magnitude,
            np.interp([-1, 0.5, 3], [0, 1, 2], [0.5, 2, 5], left=1, right=3),
        )


class TestScaledDimensionlessStructural:
    """Issue #5: structural numpy functions taking a bare value must treat it as
    a dimensionless (base) value for a dimensionless-scaled quantity."""

    @staticmethod
    def _scaled(ureg: UnitRegistry) -> ArrayQuantity:
        return ArrayQuantity(
            np.array([0.05, 0.2, 0.5]), "cm*s/m/ms", ureg
        )  # base [0.5,2,5]

    def test_full_like_append_pad(self, ureg: UnitRegistry) -> None:
        a = self._scaled(ureg)
        np.testing.assert_allclose(
            np.full_like(a, 2).to_base_units().magnitude, [2.0, 2.0, 2.0]
        )
        np.testing.assert_allclose(
            np.append(a, 2).to_base_units().magnitude, [0.5, 2.0, 5.0, 2.0]
        )
        np.testing.assert_allclose(
            np.pad(a, (1, 1), constant_values=2).to_base_units().magnitude,
            [2.0, 0.5, 2.0, 5.0, 2.0],
        )

    def test_cumprod_base(self, ureg: UnitRegistry) -> None:
        a = self._scaled(ureg)
        np.testing.assert_allclose(
            np.cumprod(a).to_base_units().magnitude, [0.5, 1.0, 5.0]
        )

    def test_searchsorted_put(self, ureg: UnitRegistry) -> None:
        assert self._scaled(ureg).searchsorted(2) == 1
        a = self._scaled(ureg)
        a.put([0, 2], 2)
        np.testing.assert_allclose(a.to_base_units().magnitude, [2.0, 2.0, 2.0])

    def test_concatenate_bare_array(self, ureg: UnitRegistry) -> None:
        a = self._scaled(ureg)
        result = np.concatenate([a, np.array([2.0])]).to_base_units()
        np.testing.assert_allclose(result.magnitude, [0.5, 2.0, 5.0, 2.0])


class TestScaledDimensionlessCreationClip:
    """Issue #5: array-creation funcs and the scalar clip() method must treat a
    bare value as dimensionless (base) for a dimensionless-scaled quantity."""

    def test_linspace_mixed_endpoints(self, ureg: UnitRegistry) -> None:
        start = ureg.Quantity(100.0, "millisecond/second")  # base 0.1
        result = np.linspace(start, 1, 3).to_base_units()
        np.testing.assert_allclose(result.magnitude, [0.1, 0.55, 1.0])

    def test_stack_bare_array(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(np.array([100.0, 200.0]), "millisecond/second")
        result = np.stack([a, [1, 2]]).to_base_units()
        np.testing.assert_allclose(result.magnitude, [[0.1, 0.2], [1.0, 2.0]])

    def test_scalar_clip_method(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(200.0, "millisecond/second")  # base 0.2
        assert q.clip(0.5, 1.5).to_base_units().magnitude == pytest.approx(0.5)
        q2 = ureg.Quantity(800.0, "millisecond/second")  # base 0.8
        assert q2.clip(0.5, 1.5).to_base_units().magnitude == pytest.approx(0.8)


class TestScaledDimensionlessClipFillMethods:
    """Issue #5: ArrayQuantity.clip()/fill() methods must treat bare values as
    dimensionless (base) for a dimensionless-scaled quantity."""

    @staticmethod
    def _scaled(ureg: UnitRegistry) -> ArrayQuantity:
        return ArrayQuantity(np.array([0.05, 0.2, 0.5]), "cm*s/m/ms", ureg)

    def test_clip_method(self, ureg: UnitRegistry) -> None:
        a = self._scaled(ureg)
        np.testing.assert_allclose(a.clip(1, 3).to_base_units().magnitude, [1, 2, 3])

    def test_fill_method(self, ureg: UnitRegistry) -> None:
        a = self._scaled(ureg)
        a.fill(2)
        np.testing.assert_allclose(a.to_base_units().magnitude, [2, 2, 2])

    def test_dimensional_clip_method_raises(self, ureg: UnitRegistry) -> None:
        # Dimensional clip with bare bounds raises, like pint.
        m = ArrayQuantity(np.array([1.0, 3.0, 5.0]), "meter", ureg)
        with pytest.raises(DimensionalityError):
            _ = m.clip(2, 4)


class TestScaledDimensionlessSetitemDot:
    """Issue #5: __setitem__ and dot must treat bare values as dimensionless
    (base) for a dimensionless-scaled quantity."""

    def test_setitem_public_rust_path(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(np.array([1000.0, 2000.0, 3000.0]), "ms/s")  # base [1,2,3]
        q[0] = 2
        np.testing.assert_allclose(q.to_base_units().magnitude, [2.0, 2.0, 3.0])
        q[:2] = [2, 3]
        np.testing.assert_allclose(q.to_base_units().magnitude, [2.0, 3.0, 3.0])

    def test_setitem_python_arrayquantity(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([1000.0, 2000.0, 3000.0]), "ms/s", ureg)
        a[0] = 2
        np.testing.assert_allclose(a.to_base_units().magnitude, [2.0, 2.0, 3.0])

    def test_dot_bare_list_tracks_units(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([1000.0, 2000.0, 3000.0]), "ms/s", ureg)
        result = a.dot([1, 1, 1])
        assert result.to_base_units().magnitude == pytest.approx(6.0)


class TestScalarQuantityPlusArray:
    """Issue #5: scalar Quantity combined with a bare ndarray must convert a
    dimensionless-scaled quantity to base units."""

    def test_scalar_plus_ndarray(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1000, "ms/s")  # base 1
        np.testing.assert_allclose(
            (q + np.array([1.0, 2.0])).to_base_units().magnitude, [2.0, 3.0]
        )
        np.testing.assert_allclose(
            (np.array([1.0, 2.0]) - q).to_base_units().magnitude, [0.0, 1.0]
        )

    def test_scalar_dimensional_plus_ndarray_raises(self, ureg: UnitRegistry) -> None:
        with pytest.raises(DimensionalityError):
            _ = ureg.Quantity(5, "meter") + np.array([1.0, 2.0])
        # Adding zeros is allowed for any dimension.
        np.testing.assert_allclose(
            (ureg.Quantity(5, "meter") + np.zeros(2)).magnitude, [5.0, 5.0]
        )


class TestScaledDimensionlessEquality:
    """Issue #5: == / != against a bare value compare base units."""

    def test_eq_bare_base(self, ureg: UnitRegistry) -> None:
        # base [0.5, 2, 5], raw stored [500, 2000, 5000]
        for q in (
            ureg.Quantity(np.array([500.0, 2000.0, 5000.0]), "ms/s"),
            ArrayQuantity(np.array([500.0, 2000.0, 5000.0]), "ms/s", ureg),
        ):
            np.testing.assert_array_equal(q == [0.5, 2.0, 5.0], [True, True, True])
            np.testing.assert_array_equal(
                q == [500.0, 2000.0, 5000.0], [False, False, False]
            )
            np.testing.assert_array_equal(
                q != [500.0, 2000.0, 5000.0], [True, True, True]
            )


class TestNewlyImplementedOps:
    """Operations pint supports that pintrs previously raised TypeError on."""

    def test_reverse_pow(self, ureg: UnitRegistry) -> None:
        assert 2 ** ureg.Quantity(2.0, "") == pytest.approx(4.0)
        # dimensionless-scaled exponent uses its base value
        assert 2 ** ureg.Quantity(2000.0, "ms/s") == pytest.approx(4.0)
        with pytest.raises(DimensionalityError):
            _ = 2 ** ureg.Quantity(2.0, "meter")

    def test_average(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([1.0, 2.0, 3.0]), "meter", ureg)
        r = np.average(a, weights=[1, 1, 2])
        assert r.magnitude == pytest.approx(2.25)
        assert str(r.units) == "meter"

    def test_insert(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([1.0, 2.0, 3.0]), "meter", ureg)
        r = np.insert(a, 1, ureg.Quantity(2.0, "meter"))
        np.testing.assert_allclose(r.magnitude, [1.0, 2.0, 2.0, 3.0])
        with pytest.raises(DimensionalityError):
            _ = np.insert(a, 1, 2)  # bare into dimensional

    def test_nan_to_num(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([1.0, np.nan, 3.0]), "meter", ureg)
        np.testing.assert_allclose(np.nan_to_num(a).magnitude, [1.0, 0.0, 3.0])

    def test_fix(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(np.array([2.7, -2.7]), "meter", ureg)
        np.testing.assert_allclose(np.fix(a).magnitude, [2.0, -2.0])


class TestAngularConversionUfuncs:
    """np.radians/degrees/deg2rad/rad2deg as angular conversions (pint parity)."""

    def test_radians(self, ureg: UnitRegistry) -> None:
        r = np.radians(ureg.Quantity(np.array([180.0]), "degree"))
        np.testing.assert_allclose(r.magnitude, [np.pi])
        assert str(r.units) == "radian"

    def test_degrees(self, ureg: UnitRegistry) -> None:
        r = np.degrees(ureg.Quantity(np.array([np.pi]), "radian"))
        np.testing.assert_allclose(r.magnitude, [180.0])
        assert str(r.units) == "degree"

    def test_dimensionless_treated_as_radians(self, ureg: UnitRegistry) -> None:
        r = np.degrees(ureg.Quantity(np.array([np.pi]), ""))
        np.testing.assert_allclose(r.magnitude, [180.0])

    def test_dimensional_raises(self, ureg: UnitRegistry) -> None:
        with pytest.raises(DimensionalityError):
            _ = np.radians(ureg.Quantity(np.array([1.0]), "meter"))


class TestMoreNumpyFunctions:
    """Further numpy functions brought to pint parity."""

    def test_conj_preserves_complex(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(np.array([1 + 2j, 3 - 4j]), "meter")
        np.testing.assert_array_equal(np.conj(q).magnitude, [1 - 2j, 3 + 4j])

    def test_prod_powers_units(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(np.array([1.0, 2.0, 3.0]), "meter")
        r = np.prod(q)
        assert r.magnitude == pytest.approx(6.0)
        assert "meter ** 3" in str(r.units).replace("meter**3", "meter ** 3")

    def test_allclose_quantity_atol(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(np.array([1.0, 2.0]), "meter")
        b = ureg.Quantity(np.array([1.01, 2.01]), "meter")
        assert np.allclose(a, b, atol=ureg.Quantity(2.0, "centimeter"))
        assert not np.allclose(a, b, atol=ureg.Quantity(0.1, "centimeter"))

    def test_unwrap(self, ureg: UnitRegistry) -> None:
        r = np.unwrap(ureg.Quantity(np.array([0.0, 4.0, 8.0]), "radian"))
        assert str(r.units) == "radian"

    def test_searchsorted_free_converts(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(np.array([100.0, 200.0, 400.0]), "centimeter")
        r = np.searchsorted(a, ureg.Quantity(np.array([2.5]), "meter"))
        np.testing.assert_array_equal(r, [2])

    def test_trapezoid_quantity_dx(self, ureg: UnitRegistry) -> None:
        trapz = getattr(np, "trapezoid", None) or np.trapz
        r = trapz(
            ureg.Quantity(np.array([1.0, 2.0, 3.0]), "meter"),
            dx=ureg.Quantity(2.0, "second"),
        )
        assert r.magnitude == pytest.approx(8.0)
        units_str = str(r.units)
        assert "meter" in units_str
        assert "second" in units_str


class TestEinsumAndComplex:
    def test_einsum_product_units(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(np.array([1.0, 2.0, 3.0]), "meter")
        b = ureg.Quantity(np.array([4.0, 5.0, 6.0]), "second")
        r = np.einsum("i,i->", a, b)
        assert r.magnitude == pytest.approx(32.0)
        units_str = str(r.units)
        assert "meter" in units_str
        assert "second" in units_str

    def test_complex_array_quantity_abs(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(np.array([3 + 4j]), "meter")
        np.testing.assert_allclose(np.abs(q).magnitude, [5.0])


class TestArrayOffsetLogConversion:
    """Array .to/.to_base_units apply the offset/log transform element-wise,
    not a bare multiplicative factor (regression for degC/dB arrays)."""

    def test_array_degc_to_kelvin(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(np.array([0.0, 10.0]), "degC").to("kelvin")
        np.testing.assert_allclose(q.magnitude, [273.15, 283.15])

    def test_array_degc_to_base_units(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(np.array([0.0, 10.0]), "degC").to_base_units()
        np.testing.assert_allclose(q.magnitude, [273.15, 283.15])

    def test_array_dbw_to_watt(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(np.array([0.0, 10.0]), "dBW").to("watt")
        np.testing.assert_allclose(q.magnitude, [1.0, 10.0])

    def test_python_array_quantity_degc(self, ureg: UnitRegistry) -> None:
        # 2D forces the pure-Python ArrayQuantity backend.
        aq = ArrayQuantity(np.array([[0.0, 10.0]]), "degC", ureg)
        np.testing.assert_allclose(aq.to("kelvin").magnitude, [[273.15, 283.15]])

    def test_array_ordinary_unit_unaffected(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(np.array([1.0, 2.0]), "km").to("m")
        np.testing.assert_allclose(q.magnitude, [1000.0, 2000.0])


class TestOffsetUnitNdarrayDirectional:
    """pint forbids offset-unit `*`/`/` only when the offset operand is on the
    left (the operation is initiated on it); when a bare ndarray is on the left
    the numpy ufunc path keeps it elementwise. pintrs matches that asymmetry."""

    def test_offset_on_left_raises(self, ureg: UnitRegistry) -> None:
        from pintrs import OffsetUnitCalculusError

        nd = np.array([1.0, 2.0])
        with pytest.raises(OffsetUnitCalculusError):
            _ = ureg.Quantity(10.0, "degC") * nd
        with pytest.raises(OffsetUnitCalculusError):
            _ = ureg.Quantity(10.0, "degC") / nd
        with pytest.raises(OffsetUnitCalculusError):
            _ = ureg.Unit("degC") * nd
        with pytest.raises(OffsetUnitCalculusError):
            _ = ureg.Quantity(np.array([1.0, 2.0]), "meter") / ureg.Unit("degC")
        with pytest.raises(OffsetUnitCalculusError):
            _ = ureg.Quantity(2.0, "degC") / ureg.Quantity(np.array([1.0, 2.0]), "m")

    def test_ndarray_on_left_allowed(self, ureg: UnitRegistry) -> None:
        nd = np.array([1.0, 2.0])
        np.testing.assert_allclose(
            (nd * ureg.Quantity(10.0, "degC")).magnitude, [10, 20]
        )
        np.testing.assert_allclose(
            (nd / ureg.Quantity(10.0, "degC")).magnitude, [0.1, 0.2]
        )
        np.testing.assert_allclose((nd * ureg.Unit("degC")).magnitude, [1, 2])
        np.testing.assert_allclose((ureg.Unit("degC") / nd).magnitude, [1, 0.5])

    def test_ordinary_unit_ndarray_unaffected(self, ureg: UnitRegistry) -> None:
        nd = np.array([1.0, 2.0])
        np.testing.assert_allclose((ureg.Quantity(10.0, "m") * nd).magnitude, [10, 20])
        np.testing.assert_allclose((ureg.Quantity(10.0, "m") / nd).magnitude, [10, 5])
        np.testing.assert_allclose((ureg.Unit("m") * nd).magnitude, [1, 2])

    def test_list_left_offset_raises(self, ureg: UnitRegistry) -> None:
        # A list/tuple on the left uses ordinary operator dispatch (not the numpy
        # ufunc path), so the offset rule applies and pint raises.
        from pintrs import OffsetUnitCalculusError

        with pytest.raises(OffsetUnitCalculusError):
            _ = [1, 2] * ureg.Quantity(10.0, "degC")
        with pytest.raises(OffsetUnitCalculusError):
            _ = [1, 2] / ureg.Quantity(10.0, "degC")
        with pytest.raises(OffsetUnitCalculusError):
            _ = (1, 2) * ureg.Quantity(10.0, "degC")
        with pytest.raises(OffsetUnitCalculusError):
            _ = [1, 2] * ureg.Unit("degC")

    def test_list_left_ordinary_unaffected(self, ureg: UnitRegistry) -> None:
        got = ([1, 2] * ureg.Quantity(10.0, "m")).magnitude
        np.testing.assert_allclose(got, [10, 20])
        got = ([1, 2] / ureg.Quantity(10.0, "m")).magnitude
        np.testing.assert_allclose(got, [0.1, 0.2])

    def test_np_power_offset_base_raises(self, ureg: UnitRegistry) -> None:
        # np.power mirrors the scalar `**` rule: an offset/log base to a power
        # other than 0 or 1 is ambiguous.
        from pintrs import OffsetUnitCalculusError

        with pytest.raises(OffsetUnitCalculusError):
            _ = np.power(ureg.Quantity(10.0, "degC"), 2)
        with pytest.raises(OffsetUnitCalculusError):
            _ = np.power(ureg.Quantity(np.array([1.0, 2.0]), "degC"), 2)
        with pytest.raises(OffsetUnitCalculusError):
            _ = np.power(ureg.Quantity(3.0, "dB"), 2)
        assert np.power(ureg.Quantity(10.0, "degC"), 1).magnitude == pytest.approx(10.0)
        assert np.power(ureg.Quantity(10.0, "degC"), 0).magnitude == pytest.approx(1.0)
        assert np.power(ureg.Quantity(10.0, "m"), 2).magnitude == pytest.approx(100.0)


class TestArrayHandlerOffsetLog:
    """Free-function handlers that combine arrays into one unit apply the
    offset/log transform (32 degF -> 0 degC), matching pint."""

    def test_concatenate_offset(self, ureg: UnitRegistry) -> None:
        r = np.concatenate(
            [
                ureg.Quantity(np.array([0.0]), "degC"),
                ureg.Quantity(np.array([32.0]), "degF"),
            ]
        )
        # 32 degF == 0 degC (atol covers the offset-roundtrip fp noise).
        np.testing.assert_allclose(r.magnitude, [0.0, 0.0], atol=1e-9)

    def test_append_log(self, ureg: UnitRegistry) -> None:
        r = np.concatenate(
            [
                ureg.Quantity(np.array([0.0]), "dBW"),
                ureg.Quantity(np.array([10.0]), "watt"),
            ]
        )
        np.testing.assert_allclose(r.magnitude, [0.0, 10.0])

    def test_insert_offset(self, ureg: UnitRegistry) -> None:
        r = np.insert(
            ureg.Quantity(np.array([0.0, 100.0]), "degC"),
            1,
            ureg.Quantity(32.0, "degF"),
        )
        np.testing.assert_allclose(r.magnitude, [0.0, 0.0, 100.0], atol=1e-9)

    def test_ordinary_handler_unaffected(self, ureg: UnitRegistry) -> None:
        r = np.concatenate(
            [ureg.Quantity(np.array([1.0]), "m"), ureg.Quantity(np.array([2.0]), "km")]
        )
        np.testing.assert_allclose(r.magnitude, [1.0, 2000.0])


class TestArrayAdditiveOffsetCalculus:
    """Array `+`/`-` follow pint's offset/log calculus: abs+abs raises,
    abs-abs yields a delta, abs+delta stays absolute."""

    def test_offset_plus_offset_raises(self, ureg: UnitRegistry) -> None:
        from pintrs import OffsetUnitCalculusError

        a = ureg.Quantity(np.array([10.0, 20.0]), "degC")
        with pytest.raises(OffsetUnitCalculusError):
            _ = a + a

    def test_log_plus_log_raises(self, ureg: UnitRegistry) -> None:
        from pintrs import OffsetUnitCalculusError

        a = ureg.Quantity(np.array([10.0]), "dB")
        with pytest.raises(OffsetUnitCalculusError):
            _ = a + a

    def test_offset_minus_offset_is_delta(self, ureg: UnitRegistry) -> None:
        r = ureg.Quantity(np.array([10.0]), "degC") - ureg.Quantity(
            np.array([5.0]), "degC"
        )
        np.testing.assert_allclose(r.magnitude, [5.0])
        assert "delta" in str(r.units)

    def test_offset_plus_delta_stays_absolute(self, ureg: UnitRegistry) -> None:
        r = ureg.Quantity(np.array([10.0]), "degC") + ureg.Quantity(
            np.array([5.0]), "delta_degC"
        )
        np.testing.assert_allclose(r.magnitude, [15.0])
        assert str(r.units) == "degree_Celsius"
