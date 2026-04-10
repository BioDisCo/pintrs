"""Tests for GitHub issue fixes.

Issue #1: pi_theorem should accept Quantity values, not just strings.
Issue #2: Assignment to numpy array elements via ureg.Quantity (RustArrayQuantity).
"""

from __future__ import annotations

import numpy as np
import pytest
from pintrs import DimensionalityError, UnitRegistry


@pytest.fixture
def ureg() -> UnitRegistry:
    return UnitRegistry()


class TestPiTheoremWithQuantities:
    """Issue #1: pi_theorem should accept dict values that are Quantity objects."""

    def test_quantity_values(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity
        result = ureg.pi_theorem(
            {
                "V": q(1, "meter/second"),
                "T": q(1, "second"),
                "L": q(1, "meter"),
            }
        )
        assert len(result) == 1
        group = result[0]
        assert group["V"] == pytest.approx(1.0)
        assert group["T"] == pytest.approx(1.0)
        assert group["L"] == pytest.approx(-1.0)

    def test_mixed_str_and_quantity(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity
        result = ureg.pi_theorem(
            {
                "V": q(1, "m/s"),
                "T": "s",
                "L": "m",
            }
        )
        assert len(result) == 1

    def test_quantity_magnitude_ignored(self, ureg: UnitRegistry) -> None:
        """The magnitude of the Quantity should not affect the result."""
        q = ureg.Quantity
        result1 = ureg.pi_theorem(
            {
                "V": q(1, "m/s"),
                "T": q(1, "s"),
                "L": q(1, "m"),
            }
        )
        result2 = ureg.pi_theorem(
            {
                "V": q(100, "m/s"),
                "T": q(0.001, "s"),
                "L": q(42, "m"),
            }
        )
        assert len(result1) == len(result2)
        for g1, g2 in zip(result1, result2, strict=True):
            for key in g1:
                assert g1[key] == pytest.approx(g2[key])


class TestArraySetitemUnitConversion:
    """Issue #2: array element assignment should convert units."""

    def test_scalar_assign_same_units(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity
        x = q(np.array([0.0, 1.0]), "m/s")
        x[0] = q(5.0, "m/s")
        assert x.magnitude[0] == pytest.approx(5.0)

    def test_scalar_assign_compatible_units(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity
        x = q(np.array([0.0, 1.0]), "m/s")
        x[0] = q(3.6, "km/hr")
        assert x.magnitude[0] == pytest.approx(1.0)

    def test_scalar_assign_incompatible_units(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity
        x = q(np.array([0.0, 1.0]), "m/s")
        with pytest.raises(DimensionalityError):
            x[0] = q(1.0, "kg")

    def test_array_assign_same_units(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity
        x = q(np.array([[0.0], [1.0]]), "m/s")
        x[0] = q(np.array([5.0]), "m/s")
        assert x.magnitude[0, 0] == pytest.approx(5.0)

    def test_array_assign_compatible_units(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity
        x = q(np.array([[0.0], [1.0]]), "m/s")
        x[0] = q(np.array([3.6]), "km/hr")
        assert x.magnitude[0, 0] == pytest.approx(1.0)

    def test_array_assign_incompatible_units(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity
        x = q(np.array([[0.0], [1.0]]), "m/s")
        with pytest.raises(DimensionalityError):
            x[0] = q(np.array([1.0]), "kg")

    def test_raw_float_assign(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity
        x = q(np.array([0.0, 1.0]), "m/s")
        x[0] = 5.0
        assert x.magnitude[0] == pytest.approx(5.0)

    def test_slice_assign(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity
        x = q(np.array([0.0, 1.0, 2.0]), "m")
        x[0:2] = q(np.array([10.0, 20.0]), "m")
        np.testing.assert_array_almost_equal(x.magnitude, [10.0, 20.0, 2.0])


class TestArrayGetitem:
    """Issue #2: RustArrayQuantity should support indexing."""

    def test_getitem_scalar(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity
        x = q(np.array([1.0, 2.0, 3.0]), "meter")
        elem = x[1]
        assert elem.magnitude == pytest.approx(2.0)
        assert "meter" in str(elem.units) or str(elem.units) == "m"

    def test_getitem_slice(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity
        x = q(np.array([1.0, 2.0, 3.0, 4.0]), "meter")
        sliced = x[1:3]
        np.testing.assert_array_almost_equal(sliced.magnitude, [2.0, 3.0])


class TestFacetsCompat:
    """Facets module compatibility for pint drop-in usage."""

    def test_import_plain_quantity(self) -> None:
        from pintrs.facets.plain import PlainQuantity

        assert PlainQuantity is not None

    def test_import_plain_unit(self) -> None:
        from pintrs.facets.plain import PlainUnit

        assert PlainUnit is not None

    def test_import_numpy_quantity(self) -> None:
        from pintrs.facets.numpy.quantity import NumpyQuantity

        assert NumpyQuantity is not None

    def test_plain_quantity_isinstance_scalar(self, ureg: UnitRegistry) -> None:
        from pintrs.facets.plain import PlainQuantity

        q = ureg.Quantity(1.0, "meter")
        assert isinstance(q, PlainQuantity)

    def test_plain_quantity_isinstance_array(self, ureg: UnitRegistry) -> None:
        from pintrs.facets.plain import PlainQuantity

        q = ureg.Quantity(np.array([1.0, 2.0]), "meter")
        assert isinstance(q, PlainQuantity)

    def test_numpy_quantity_isinstance_scalar(self, ureg: UnitRegistry) -> None:
        from pintrs.facets.numpy.quantity import NumpyQuantity

        q = ureg.Quantity(1.0, "meter")
        assert isinstance(q, NumpyQuantity)

    def test_numpy_quantity_isinstance_array(self, ureg: UnitRegistry) -> None:
        from pintrs.facets.numpy.quantity import NumpyQuantity

        q = ureg.Quantity(np.array([1.0, 2.0]), "meter")
        assert isinstance(q, NumpyQuantity)

    def test_plain_quantity_generic_subscript(self) -> None:
        from pintrs.facets.plain import PlainQuantity

        alias = PlainQuantity[float]
        assert alias is PlainQuantity


class TestDimensionalityErrorCompat:
    """DimensionalityError should support pint's structured constructor."""

    def test_keyword_args(self) -> None:
        e = DimensionalityError("meter", "second", extra_msg=" test")
        assert str(e) == "Cannot convert from 'meter' to 'second' test"
        assert e.units1 == "meter"
        assert e.units2 == "second"
        assert e.extra_msg == " test"

    def test_with_dims(self) -> None:
        e = DimensionalityError("meter", "second", dim1="[length]", dim2="[time]")
        assert "([length])" in str(e)
        assert "([time])" in str(e)

    def test_catches_rust_raised(self, ureg: UnitRegistry) -> None:
        with pytest.raises(DimensionalityError):
            ureg.Quantity(1, "meter").to("second")

    def test_rust_raised_str_preserved(self, ureg: UnitRegistry) -> None:
        with pytest.raises(DimensionalityError, match="Cannot convert"):
            ureg.Quantity(1, "meter").to("second")


class TestArrayQuantityCopy:
    """Quantity.copy() support for both Rust and Python array quantities."""

    def test_rust_array_copy(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(np.array([1.0, 2.0, 3.0]), "m/s")
        c = q.copy()
        np.testing.assert_array_equal(c.magnitude, q.magnitude)
        c[0] = 99.0
        assert q.magnitude[0] == pytest.approx(1.0)

    def test_python_array_copy(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(np.array([[1.0], [2.0]]), "m/s")
        c = q.copy()
        np.testing.assert_array_equal(c.magnitude, q.magnitude)
        c.magnitude[0, 0] = 99.0
        assert q.magnitude[0, 0] == pytest.approx(1.0)
