"""Testing utilities tests - ported from pint's test_testing.py."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from pintrs import UnitRegistry
from pintrs.testing import assert_allclose, assert_equal


@pytest.fixture
def ureg() -> UnitRegistry:
    return UnitRegistry()


class TestAssertEqual:
    def test_equal_quantities(self, ureg: UnitRegistry) -> None:
        q1 = ureg.Quantity(1, "meter")
        q2 = ureg.Quantity(1, "meter")
        assert_equal(q1, q2)

    def test_equal_after_conversion(self, ureg: UnitRegistry) -> None:
        q1 = ureg.Quantity(1, "meter")
        q2 = ureg.Quantity(100, "centimeter")
        assert_equal(q1, q2)

    def test_unequal_magnitudes_raises(self, ureg: UnitRegistry) -> None:
        q1 = ureg.Quantity(1, "meter")
        q2 = ureg.Quantity(2, "meter")
        with pytest.raises(AssertionError):
            assert_equal(q1, q2)

    def test_unequal_units_raises(self, ureg: UnitRegistry) -> None:
        q1 = ureg.Quantity(1, "meter")
        q2 = ureg.Quantity(1, "second")
        with pytest.raises(AssertionError):
            assert_equal(q1, q2)

    def test_equal_dimensionless(self, ureg: UnitRegistry) -> None:
        q1 = ureg.Quantity(1, "dimensionless")
        q2 = ureg.Quantity(1, "dimensionless")
        assert_equal(q1, q2)


class TestAssertAllclose:
    def test_close_quantities(self, ureg: UnitRegistry) -> None:
        q1 = ureg.Quantity(1.0, "meter")
        q2 = ureg.Quantity(1.0 + 1e-10, "meter")
        assert_allclose(q1, q2)

    def test_close_after_conversion(self, ureg: UnitRegistry) -> None:
        q1 = ureg.Quantity(1.0, "meter")
        q2 = ureg.Quantity(100.0, "centimeter")
        assert_allclose(q1, q2)

    def test_not_close_raises(self, ureg: UnitRegistry) -> None:
        q1 = ureg.Quantity(1.0, "meter")
        q2 = ureg.Quantity(2.0, "meter")
        with pytest.raises(AssertionError):
            assert_allclose(q1, q2)

    def test_custom_rtol(self, ureg: UnitRegistry) -> None:
        q1 = ureg.Quantity(1.0, "meter")
        q2 = ureg.Quantity(1.05, "meter")
        assert_allclose(q1, q2, rtol=0.1)

    def test_custom_atol(self, ureg: UnitRegistry) -> None:
        q1 = ureg.Quantity(1.0, "meter")
        q2 = ureg.Quantity(1.001, "meter")
        assert_allclose(q1, q2, atol=0.01)
