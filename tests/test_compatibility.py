"""Compatibility tests verifying pintrs matches pint's API surface."""

from __future__ import annotations

import copy
import pickle

import pytest
from pintrs import (
    DimensionalityError,
    Measurement,
    PintError,
    Quantity,
    UndefinedUnitError,
    Unit,
    UnitRegistry,
    check,
    wraps,
)


@pytest.fixture
def ureg() -> UnitRegistry:
    return UnitRegistry()


class TestExceptionHierarchy:
    def test_dimensionality_error_is_value_error(self, ureg: UnitRegistry) -> None:
        with pytest.raises(ValueError):
            ureg.Quantity(1, "meter").to("second")

    def test_dimensionality_error_is_pint_error(self, ureg: UnitRegistry) -> None:
        with pytest.raises(PintError):
            ureg.Quantity(1, "meter").to("second")

    def test_dimensionality_error_type(self, ureg: UnitRegistry) -> None:
        with pytest.raises(DimensionalityError):
            ureg.Quantity(1, "meter").to("second")

    def test_undefined_unit_error(self, ureg: UnitRegistry) -> None:
        with pytest.raises(UndefinedUnitError):
            ureg.Quantity(1, "foobar_unit")


class TestQuantityConstruction:
    def test_from_quantity_no_convert(self, ureg: UnitRegistry) -> None:
        q1 = ureg.Quantity(5, "meter")
        q2 = ureg.Quantity(q1)
        assert q2.magnitude == 5
        assert str(q2.units) == "m"

    def test_from_quantity_convert(self, ureg: UnitRegistry) -> None:
        q1 = ureg.Quantity(1, "kilometer")
        q2 = ureg.Quantity(q1, "meter")
        assert abs(q2.magnitude - 1000) < 1e-10

    def test_from_string(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity("3.5 meters")
        assert abs(q.magnitude - 3.5) < 1e-10

    def test_from_number_and_unit(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(9.8, "m/s^2")
        assert abs(q.magnitude - 9.8) < 1e-10


class TestRegistryFeatures:
    def test_contains(self, ureg: UnitRegistry) -> None:
        assert "meter" in ureg
        assert "foobar" not in ureg

    def test_iter(self, ureg: UnitRegistry) -> None:
        names = list(ureg)
        assert len(names) > 100
        assert "meter" in names

    def test_define(self, ureg: UnitRegistry) -> None:
        ureg.define("smoot = 1.7018 * meter")
        q = ureg.Quantity(1, "smoot").to("meter")
        assert abs(q.magnitude - 1.7018) < 1e-10

    def test_convert(self, ureg: UnitRegistry) -> None:
        val = ureg.convert(1.0, "km", "meter")
        assert abs(val - 1000) < 1e-10

    def test_parse_units(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("m/s")
        assert str(u) != ""

    def test_parse_expression(self, ureg: UnitRegistry) -> None:
        q = ureg.parse_expression("9.8 m/s^2")
        assert abs(q.magnitude - 9.8) < 1e-10

    def test_getattr_unit(self, ureg: UnitRegistry) -> None:
        u = ureg.meter
        assert isinstance(u, Unit)
        assert str(u) == "m"

    def test_get_compatible_units(self, ureg: UnitRegistry) -> None:
        units = ureg.get_compatible_units("meter")
        assert "foot" in units or "inch" in units or len(units) > 5


class TestQuantityProperties:
    def test_magnitude_aliases(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5, "meter")
        assert q.magnitude == q.m

    def test_units_aliases(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5, "meter")
        assert str(q.units) == str(q.u)

    def test_dimensionality(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "meter")
        assert q.dimensionality == "[length]"

    def test_dimensionless(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "meter") / ureg.Quantity(1, "meter")
        assert q.dimensionless
        assert q.unitless

    def test_not_dimensionless(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "meter")
        assert not q.dimensionless
        assert not q.unitless


class TestConversionMethods:
    def test_to(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "km").to("meter")
        assert abs(q.magnitude - 1000) < 1e-10

    def test_ito(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "km")
        q.ito("meter")
        assert abs(q.magnitude - 1000) < 1e-10

    def test_m_as(self, ureg: UnitRegistry) -> None:
        assert abs(ureg.Quantity(1, "km").m_as("meter") - 1000) < 1e-10

    def test_to_base_units(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "km").to_base_units()
        assert abs(q.magnitude - 1000) < 1e-10

    def test_to_root_units(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "km").to_root_units()
        assert abs(q.magnitude - 1000) < 1e-10

    def test_ito_base_units(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "km")
        q.ito_base_units()
        assert abs(q.magnitude - 1000) < 1e-10

    def test_to_compact(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(0.001, "meter").to_compact()
        assert abs(q.magnitude - 1.0) < 1e-10
        assert "milli" in str(q) or "mm" in str(q)

    def test_to_compact_large(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1500, "meter").to_compact()
        assert abs(q.magnitude - 1.5) < 1e-10

    def test_to_reduced_units(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5, "km").to_reduced_units()
        assert q.magnitude > 0


class TestCheckAndCompatible:
    def test_check_true(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(1, "meter").check("[length]")

    def test_check_false(self, ureg: UnitRegistry) -> None:
        assert not ureg.Quantity(1, "meter").check("[time]")

    def test_is_compatible_with_str(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "meter")
        assert q.is_compatible_with("foot")
        assert not q.is_compatible_with("second")

    def test_compatible_units(self, ureg: UnitRegistry) -> None:
        units = ureg.Quantity(1, "meter").compatible_units()
        assert len(units) > 5


class TestArithmeticExtended:
    def test_round(self, ureg: UnitRegistry) -> None:
        q = round(ureg.Quantity(3.14159, "meter"), 2)
        assert abs(q.magnitude - 3.14) < 1e-10

    def test_ne(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(1, "meter") != ureg.Quantity(2, "meter")

    def test_float_dimensionless(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3, "meter") / ureg.Quantity(1, "meter")
        assert float(q) == 3.0

    def test_int_dimensionless(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3, "meter") / ureg.Quantity(1, "meter")
        assert int(q) == 3

    def test_bool(self, ureg: UnitRegistry) -> None:
        assert bool(ureg.Quantity(1, "meter"))
        assert not bool(ureg.Quantity(0, "meter"))


class TestCopyPickle:
    def test_copy(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5, "meter")
        q2 = copy.copy(q)
        assert str(q) == str(q2)

    def test_deepcopy(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5, "meter")
        q2 = copy.deepcopy(q)
        assert str(q) == str(q2)

    def test_pickle_quantity(self) -> None:
        q = Quantity(5.0, "meter")
        q2 = pickle.loads(pickle.dumps(q))
        assert abs(q2.magnitude - 5.0) < 1e-10

    def test_pickle_unit(self) -> None:
        from pintrs import Unit

        u = Unit("meter")
        u2 = pickle.loads(pickle.dumps(u))
        assert str(u) == str(u2)


class TestWrapsDecorator:
    def test_wraps_basic(self, ureg: UnitRegistry) -> None:
        @wraps(ureg, ret="meter", args=("meter", "meter"))
        def add_lengths(a: float, b: float) -> float:
            return a + b

        result = add_lengths(ureg.Quantity(1, "km"), ureg.Quantity(500, "meter"))
        assert abs(result.magnitude - 1500) < 1e-10
        assert str(result.units) == "m"

    def test_wraps_no_ret(self, ureg: UnitRegistry) -> None:
        @wraps(ureg, ret=None, args=("meter",))
        def get_magnitude(a: float) -> float:
            return a

        result = get_magnitude(ureg.Quantity(5, "km"))
        assert result == 5000.0


class TestCheckDecorator:
    def test_check_passes(self, ureg: UnitRegistry) -> None:
        @check(ureg, "[length]", "[time]")
        def speed(d: Quantity, t: Quantity) -> Quantity:
            return d / t

        result = speed(ureg.Quantity(100, "meter"), ureg.Quantity(10, "second"))
        assert abs(result.magnitude - 10) < 1e-10

    def test_check_fails(self, ureg: UnitRegistry) -> None:
        @check(ureg, "[length]", "[time]")
        def speed(d: Quantity, t: Quantity) -> Quantity:
            return d / t

        with pytest.raises(DimensionalityError):
            speed(
                ureg.Quantity(100, "meter"),
                ureg.Quantity(10, "kilogram"),
            )


class TestMeasurement:
    def test_basic(self, ureg: UnitRegistry) -> None:
        m = Measurement(ureg.Quantity(5.0, "meter"), 0.1)
        assert m.magnitude == 5.0
        assert abs(m.rel - 0.02) < 1e-10

    def test_str(self, ureg: UnitRegistry) -> None:
        m = Measurement(ureg.Quantity(5.0, "meter"), 0.1)
        s = str(m)
        assert "5.0" in s
        assert "0.1" in s

    def test_error_as_quantity(self, ureg: UnitRegistry) -> None:
        m = Measurement(
            ureg.Quantity(5.0, "meter"),
            ureg.Quantity(10, "centimeter"),
        )
        assert abs(m.error.magnitude - 0.1) < 1e-10


class TestUnitOperations:
    def test_unit_mul_scalar(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("meter")
        q = 5.0 * u
        assert q.magnitude == 5.0

    def test_unit_div(self, ureg: UnitRegistry) -> None:
        u1 = ureg.Unit("meter")
        u2 = ureg.Unit("second")
        u3 = u1 / u2
        assert str(u3) != ""

    def test_unit_pow(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("meter")
        u2 = u**2
        assert str(u2) != ""

    def test_unit_eq(self, ureg: UnitRegistry) -> None:
        assert ureg.Unit("meter") == ureg.Unit("meter")

    def test_unit_dimensionality(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("meter")
        assert u.dimensionality == "[length]"

    def test_unit_is_compatible_with(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("meter")
        assert u.is_compatible_with("foot")
        assert not u.is_compatible_with("second")


class TestMathOperations:
    def test_sqrt_via_pow(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(4, "meter") ** 2
        result = q**0.5
        assert abs(result.magnitude - 4.0) < 1e-10

    def test_abs(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(-5, "meter")
        assert abs(q).magnitude == 5

    def test_neg(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5, "meter")
        assert (-q).magnitude == -5
