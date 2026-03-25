"""Compatibility/compat tests - ported from pint's test_compat.py.

Tests NaN handling, equality comparison, and zero detection behavior
through the quantity API rather than pint's internal compat functions.
"""

from __future__ import annotations

import math

import pytest
from pintrs import UnitRegistry


@pytest.fixture
def ureg() -> UnitRegistry:
    return UnitRegistry()


class TestQuantityNaN:
    """Test NaN behavior in quantities (mirrors pint's isnan tests)."""

    def test_nan_magnitude(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(float("nan"), "meter")
        assert math.isnan(q.magnitude)

    def test_nan_not_equal_to_itself(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(float("nan"), "meter")
        assert q != q

    def test_nan_not_equal_to_number(self, ureg: UnitRegistry) -> None:
        q_nan = ureg.Quantity(float("nan"), "meter")
        q_num = ureg.Quantity(1.0, "meter")
        assert q_nan != q_num

    def test_zero_is_not_nan(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(0.0, "meter")
        assert not math.isnan(q.magnitude)

    def test_regular_float_is_not_nan(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "meter")
        assert not math.isnan(q.magnitude)


class TestQuantityNaNNumpy:
    """Test NaN behavior with numpy arrays in quantities via ArrayQuantity."""

    def test_numpy_nan_in_array(self, ureg: UnitRegistry) -> None:
        np = pytest.importorskip("numpy")
        from pintrs import ArrayQuantity

        q = ArrayQuantity(np.array([1.0, np.nan, 3.0]), "meter", ureg)
        assert np.any(np.isnan(q.magnitude))

    def test_numpy_no_nan(self, ureg: UnitRegistry) -> None:
        np = pytest.importorskip("numpy")
        from pintrs import ArrayQuantity

        q = ArrayQuantity(np.array([1.0, 2.0, 3.0]), "meter", ureg)
        assert not np.any(np.isnan(q.magnitude))

    def test_numpy_all_nan(self, ureg: UnitRegistry) -> None:
        np = pytest.importorskip("numpy")
        from pintrs import ArrayQuantity

        q = ArrayQuantity(np.array([np.nan, np.nan]), "meter", ureg)
        assert np.all(np.isnan(q.magnitude))


class TestQuantityEquality:
    """Test equality comparison (mirrors pint's eq tests)."""

    def test_equal_same_units(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(1.0, "meter") == ureg.Quantity(1.0, "meter")

    def test_not_equal_different_magnitude(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(1.0, "meter") != ureg.Quantity(2.0, "meter")

    def test_equal_compatible_units(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(1.0, "kilometer") == ureg.Quantity(1000.0, "meter")

    def test_not_equal_different_values(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(1.0, "meter") != ureg.Quantity(1.0, "second")


class TestQuantityEqualityNumpy:
    """Test equality with numpy arrays."""

    def test_array_equal(self, ureg: UnitRegistry) -> None:
        np = pytest.importorskip("numpy")
        from pintrs import ArrayQuantity

        a = ArrayQuantity(np.array([1.0, 2.0]), "meter", ureg)
        b = ArrayQuantity(np.array([1.0, 2.0]), "meter", ureg)
        np.testing.assert_array_equal(a.magnitude, b.magnitude)

    def test_array_not_equal(self, ureg: UnitRegistry) -> None:
        np = pytest.importorskip("numpy")
        from pintrs import ArrayQuantity

        a = ArrayQuantity(np.array([1.0, 2.0]), "meter", ureg)
        b = ArrayQuantity(np.array([1.0, 3.0]), "meter", ureg)
        assert not np.array_equal(a.magnitude, b.magnitude)


class TestQuantityZero:
    """Test zero detection (mirrors pint's zero_or_nan tests)."""

    def test_zero_is_falsy(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(0, "meter")
        assert not bool(q)

    def test_nonzero_is_truthy(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "meter")
        assert bool(q)

    def test_zero_magnitude(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(0, "meter")
        assert q.magnitude == 0

    def test_nan_is_truthy(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(float("nan"), "meter")
        assert bool(q)
