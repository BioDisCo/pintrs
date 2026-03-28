"""Run a subset of pint's test patterns against pintrs.

This tests the most common pint usage patterns to verify drop-in compatibility.
We can't run pint's full test suite directly because it imports pint internals,
but we replicate the key test patterns here using pintrs.
"""

from __future__ import annotations

import copy
import datetime
import math
import pickle

import pytest
from pintrs import (
    DimensionalityError,
    Measurement,
    Quantity,
    Unit,
    UnitRegistry,
)


@pytest.fixture
def ureg() -> UnitRegistry:
    return UnitRegistry()


# === Pint test_quantity.py patterns ===


class TestQuantityCreation:
    """Mirrors pint's TestQuantity.test_quantity_creation."""

    def test_number_and_string(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(4.2, "meter")
        assert q.magnitude == 4.2

    def test_string(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity("4.2*meter")
        assert abs(q.magnitude - 4.2) < 1e-10

    def test_quantity_from_quantity(self, ureg: UnitRegistry) -> None:
        q1 = ureg.Quantity(4.2, "meter")
        q2 = ureg.Quantity(q1)
        assert q2.magnitude == 4.2

    def test_quantity_from_quantity_convert(self, ureg: UnitRegistry) -> None:
        q1 = ureg.Quantity(4.2, "meter")
        q2 = ureg.Quantity(q1, "centimeter")
        assert abs(q2.magnitude - 420.0) < 1e-10

    def test_nan(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(float("nan"), "meter")
        assert math.isnan(q.magnitude)

    def test_inf(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(float("inf"), "meter")
        assert math.isinf(q.magnitude)

    def test_empty_string_raises(self, ureg: UnitRegistry) -> None:
        with pytest.raises(ValueError):
            ureg.Quantity("")


class TestQuantityToFromTuple:
    """Mirrors pint's test_quantity_to_from_tuple."""

    def test_to_tuple(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(4.2, "meter")
        mag, items = q.to_tuple()
        assert mag == 4.2
        assert items[0][0] == "meter"
        assert items[0][1] == 1.0

    def test_from_tuple(self) -> None:
        q = Quantity.from_tuple((4.2, [("meter", 1.0)]))
        assert abs(q.magnitude - 4.2) < 1e-10


class TestQuantityConversion:
    """Mirrors pint's conversion tests."""

    def test_convert(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "inch")
        result = q.to("centimeter")
        assert abs(result.magnitude - 2.54) < 1e-6

    def test_convert_from(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(2.0, "inch")
        assert abs(q.m_as("meter") - 0.0508) < 1e-6

    def test_incompatible_raises(self, ureg: UnitRegistry) -> None:
        with pytest.raises(DimensionalityError):
            ureg.Quantity(1.0, "meter").to("second")

    def test_to_base_units(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "kilometer")
        base = q.to_base_units()
        assert abs(base.magnitude - 5000.0) < 1e-10
        assert str(base.units) == "m"

    def test_ito(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "kilometer")
        q.ito("meter")
        assert abs(q.magnitude - 1000.0) < 1e-10

    def test_to_compact(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(0.001, "meter")
        compact = q.to_compact()
        assert abs(compact.magnitude - 1.0) < 1e-10
        assert "milli" in str(compact) or "mm" in str(compact)

    def test_to_unprefixed(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5, "km")
        result = q.to_unprefixed()
        assert abs(result.magnitude - 5000) < 1e-10


class TestQuantityArithmetic:
    """Mirrors pint's arithmetic tests."""

    def test_add_same_units(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(1.0, "meter")
        b = ureg.Quantity(2.0, "meter")
        assert (a + b).magnitude == 3.0

    def test_add_compatible(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(1.0, "kilometer")
        b = ureg.Quantity(25.0, "meter")
        result = a + b
        assert abs(result.magnitude - 1.025) < 1e-10

    def test_add_incompatible(self, ureg: UnitRegistry) -> None:
        with pytest.raises(DimensionalityError):
            ureg.Quantity(1.0, "meter") + ureg.Quantity(1.0, "second")

    def test_sub(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(10.0, "meter")
        b = ureg.Quantity(3.0, "meter")
        assert (a - b).magnitude == 7.0

    def test_mul(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(2.0, "meter")
        b = ureg.Quantity(3.0, "second")
        result = a * b
        assert result.magnitude == 6.0

    def test_mul_scalar(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "meter")
        assert (q * 3).magnitude == 15.0
        assert (3 * q).magnitude == 15.0
        assert (3.0 * q).magnitude == 15.0

    def test_div(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(10.0, "meter")
        b = ureg.Quantity(2.0, "second")
        result = a / b
        assert result.magnitude == 5.0

    def test_div_scalar(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(10.0, "meter")
        assert (q / 2).magnitude == 5.0

    def test_rdiv(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(2.0, "meter")
        result = 10.0 / q
        assert result.magnitude == 5.0

    def test_pow(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.0, "meter")
        result = q**2
        assert result.magnitude == 9.0

    def test_neg(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "meter")
        assert (-q).magnitude == -5.0

    def test_abs(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(-5.0, "meter")
        assert abs(q).magnitude == 5.0

    def test_round(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.14159, "meter")
        assert abs(round(q, 2).magnitude - 3.14) < 1e-10


class TestQuantityComparison:
    """Mirrors pint's comparison tests."""

    def test_eq(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(1.0, "meter") == ureg.Quantity(1.0, "meter")

    def test_eq_different_units(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(1.0, "kilometer") == ureg.Quantity(1000.0, "meter")

    def test_ne(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(1.0, "meter") != ureg.Quantity(2.0, "meter")

    def test_lt(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(1.0, "meter") < ureg.Quantity(2.0, "meter")

    def test_gt(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(2.0, "meter") > ureg.Quantity(1.0, "meter")

    def test_le(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(1.0, "meter") <= ureg.Quantity(1.0, "meter")
        assert ureg.Quantity(1.0, "meter") <= ureg.Quantity(2.0, "meter")

    def test_ge(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(2.0, "meter") >= ureg.Quantity(2.0, "meter")
        assert ureg.Quantity(2.0, "meter") >= ureg.Quantity(1.0, "meter")

    def test_compare_method(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(1.0, "km")
        b = ureg.Quantity(500.0, "meter")
        assert a.compare(b, ">")
        assert not a.compare(b, "==")


class TestQuantityDimensionality:
    """Mirrors pint's dimensionality tests."""

    def test_dimensionality(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "meter")
        assert q.dimensionality == "[length]"
        assert not q.dimensionless

    def test_dimensionless(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "meter") / ureg.Quantity(1.0, "meter")
        assert q.dimensionless
        assert q.unitless

    def test_float_dimensionless(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.0, "meter") / ureg.Quantity(1.0, "meter")
        assert float(q) == 3.0

    def test_float_dimensioned_raises(self, ureg: UnitRegistry) -> None:
        with pytest.raises(DimensionalityError):
            float(ureg.Quantity(3.0, "meter"))

    def test_is_compatible_with(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "meter")
        assert q.is_compatible_with("foot")
        assert q.is_compatible_with("inch")
        assert not q.is_compatible_with("second")

    def test_check(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "meter")
        assert q.check("[length]")
        assert not q.check("[time]")


class TestQuantityRepr:
    """Mirrors pint's repr/str tests."""

    def test_str(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.5, "meter")
        assert str(q) == "3.5 m"

    def test_repr(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.5, "meter")
        r = repr(q)
        assert "3.5" in r
        assert "m" in r

    def test_format_empty(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.5, "meter")
        assert format(q, "") == "3.5 m"


class TestQuantityPickle:
    """Mirrors pint's pickle tests."""

    def test_pickle(self) -> None:
        q = Quantity(5.0, "meter")
        q2 = pickle.loads(pickle.dumps(q))
        assert abs(q2.magnitude - 5.0) < 1e-10
        assert str(q2.units) == "m"

    def test_copy(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "meter")
        q2 = copy.copy(q)
        assert q.magnitude == q2.magnitude

    def test_deepcopy(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "meter")
        q2 = copy.deepcopy(q)
        assert q.magnitude == q2.magnitude


class TestOffsetUnits:
    """Mirrors pint's temperature tests."""

    def test_celsius_to_kelvin(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(100, "degree_Celsius")
        result = q.to("kelvin")
        assert abs(result.magnitude - 373.15) < 1e-6

    def test_celsius_to_fahrenheit(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(100, "degree_Celsius")
        result = q.to("degree_Fahrenheit")
        assert abs(result.magnitude - 212.0) < 1e-6

    def test_fahrenheit_to_celsius(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(32, "degree_Fahrenheit")
        result = q.to("degree_Celsius")
        assert abs(result.magnitude) < 1e-6

    def test_absolute_zero(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(0, "kelvin")
        result = q.to("degree_Celsius")
        assert abs(result.magnitude - (-273.15)) < 1e-6


class TestTimedelta:
    """Mirrors pint's timedelta tests."""

    def test_to_timedelta(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(60, "second")
        td = q.to_timedelta()
        assert isinstance(td, datetime.timedelta)
        assert td.total_seconds() == 60.0

    def test_to_timedelta_minutes(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.5, "minute")
        td = q.to_timedelta()
        assert abs(td.total_seconds() - 90.0) < 1e-6

    def test_to_timedelta_hours(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "hour")
        td = q.to_timedelta()
        assert abs(td.total_seconds() - 3600.0) < 1e-6


class TestUnitOperations:
    """Mirrors pint's unit operation tests."""

    def test_unit_from(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("meter")
        q = u.from_(ureg.Quantity(1, "km"))
        assert abs(q.magnitude - 1000) < 1e-10

    def test_unit_m_from(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("meter")
        assert abs(u.m_from(ureg.Quantity(1, "km")) - 1000) < 1e-10

    def test_unit_mul(self, ureg: UnitRegistry) -> None:
        u1 = ureg.Unit("meter")
        u2 = ureg.Unit("second")
        u3 = u1 * u2
        assert str(u3) != ""

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

    def test_unit_is_compatible_with(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("meter")
        assert u.is_compatible_with("foot")
        assert not u.is_compatible_with("second")

    def test_unit_dimensionality(self, ureg: UnitRegistry) -> None:
        assert ureg.Unit("meter").dimensionality == "[length]"

    def test_unit_pickle(self) -> None:
        u = Unit("meter")
        u2 = pickle.loads(pickle.dumps(u))
        assert str(u) == str(u2)


class TestRegistryOperations:
    """Mirrors pint's registry tests."""

    def test_parse_expression(self, ureg: UnitRegistry) -> None:
        q = ureg.parse_expression("9.8 m/s^2")
        assert abs(q.magnitude - 9.8) < 1e-10

    def test_parse_units(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("m/s")
        assert str(u) != ""

    def test_getattr(self, ureg: UnitRegistry) -> None:
        u = ureg.meter
        assert isinstance(u, Unit)
        assert str(u) == "m"

    def test_contains(self, ureg: UnitRegistry) -> None:
        assert "meter" in ureg
        assert "foobar" not in ureg

    def test_iter(self, ureg: UnitRegistry) -> None:
        names = list(ureg)
        assert "meter" in names
        assert len(names) > 100

    def test_define(self, ureg: UnitRegistry) -> None:
        ureg.define("smoot = 1.7018 * meter")
        q = ureg.Quantity(1, "smoot").to("meter")
        assert abs(q.magnitude - 1.7018) < 1e-10

    def test_convert(self, ureg: UnitRegistry) -> None:
        assert abs(ureg.convert(1.0, "km", "meter") - 1000) < 1e-10

    def test_get_base_units(self, ureg: UnitRegistry) -> None:
        factor, _unit = ureg.get_base_units("kilometer")
        assert abs(factor - 1000) < 1e-10

    def test_get_name(self, ureg: UnitRegistry) -> None:
        assert ureg.get_name("m") == "meter"

    def test_get_symbol(self, ureg: UnitRegistry) -> None:
        assert ureg.get_symbol("meter") == "m"

    def test_is_compatible_with(self, ureg: UnitRegistry) -> None:
        assert ureg.is_compatible_with("meter", "foot")
        assert not ureg.is_compatible_with("meter", "second")

    def test_get_compatible_units(self, ureg: UnitRegistry) -> None:
        units = ureg.get_compatible_units("meter")
        assert len(units) > 5


class TestVariousUnits:
    """Mirrors pint's various unit conversion tests."""

    CONVERSIONS = [
        ("atmosphere", "pascal", 101325, 1),
        ("mile", "kilometer", 1.609344, 1e-4),
        ("gallon", "liter", 3.785412, 1e-3),
        ("horsepower", "watt", 745.7, 1.0),
        ("electron_volt", "joule", 1.602e-19, 1e-22),
        ("speed_of_light", "m/s", 299792458, 1),
        ("inch", "centimeter", 2.54, 1e-6),
        ("pound", "kilogram", 0.453592, 1e-3),
        ("foot", "meter", 0.3048, 1e-6),
        ("yard", "meter", 0.9144, 1e-6),
        ("acre", "m**2", 4046.86, 1),
        ("bar", "pascal", 1e5, 1),
        ("calorie", "joule", 4.184, 1e-3),
    ]

    @pytest.mark.parametrize(
        ("src", "dst", "expected", "tol"),
        CONVERSIONS,
        ids=[f"{s}->{d}" for s, d, _, _ in CONVERSIONS],
    )
    def test_conversion(
        self,
        ureg: UnitRegistry,
        src: str,
        dst: str,
        expected: float,
        tol: float,
    ) -> None:
        q = ureg.Quantity(1, src)
        result = q.to(dst)
        assert abs(result.magnitude - expected) < tol


class TestWrapsDecorator:
    """Mirrors pint's wraps tests."""

    def test_basic_wraps(self, ureg: UnitRegistry) -> None:
        @ureg.wraps("meter", ("meter", "meter"))
        def add_lengths(a: float, b: float) -> float:
            return a + b

        result = add_lengths(ureg.Quantity(1, "km"), ureg.Quantity(500, "meter"))
        assert abs(result.magnitude - 1500) < 1e-10

    def test_check_decorator(self, ureg: UnitRegistry) -> None:
        @ureg.check("[length]", "[time]")
        def speed(d: Quantity, t: Quantity) -> Quantity:
            return d / t

        result = speed(ureg.Quantity(100, "meter"), ureg.Quantity(10, "second"))
        assert abs(result.magnitude - 10) < 1e-10

    def test_check_fails(self, ureg: UnitRegistry) -> None:
        @ureg.check("[length]", "[time]")
        def speed(d: Quantity, t: Quantity) -> Quantity:
            return d / t

        with pytest.raises(DimensionalityError):
            speed(
                ureg.Quantity(100, "meter"),
                ureg.Quantity(10, "kilogram"),
            )


class TestMeasurement:
    """Mirrors pint's measurement tests."""

    def test_plus_minus(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "meter")
        m = q.plus_minus(0.1)
        assert isinstance(m, Measurement)
        assert m.magnitude == 5.0
        assert abs(m.rel - 0.02) < 1e-10

    def test_measurement_constructor(self, ureg: UnitRegistry) -> None:
        m = ureg.Measurement(5.0, 0.1, "meter")
        assert isinstance(m, Measurement)
        assert m.magnitude == 5.0

    def test_measurement_str(self, ureg: UnitRegistry) -> None:
        m = Measurement(ureg.Quantity(5.0, "meter"), 0.1)
        s = str(m)
        assert "5.0" in s
        assert "0.1" in s
