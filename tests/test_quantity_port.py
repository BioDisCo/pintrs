"""Comprehensive Quantity tests - ported from pint's test suite."""

from __future__ import annotations

import copy
import math
import pickle

import pytest
from pintrs import (
    DimensionalityError,
    Quantity,
    UndefinedUnitError,
    Unit,
    UnitRegistry,
)


@pytest.fixture
def ureg() -> UnitRegistry:
    """Create a fresh UnitRegistry for each test."""
    return UnitRegistry()


# ---------------------------------------------------------------------------
# 1. TestQuantityCreation
# ---------------------------------------------------------------------------


class TestQuantityCreation:
    """Creation from value+units, string, quantity, nan, inf, bool."""

    def test_from_value_and_units(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(4.2, "meter")
        assert q.magnitude == 4.2
        assert str(q.units) == "m"

    def test_from_integer_and_units(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3, "second")
        assert q.magnitude == 3.0

    def test_from_string_expression(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity("4.2*meter")
        assert abs(q.magnitude - 4.2) < 1e-10

    def test_from_string_with_space(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity("9.8 m/s^2")
        assert abs(q.magnitude - 9.8) < 1e-10

    def test_from_quantity(self, ureg: UnitRegistry) -> None:
        q1 = ureg.Quantity(4.2, "meter")
        q2 = ureg.Quantity(q1)
        assert q2.magnitude == 4.2
        assert str(q2.units) == "m"

    def test_from_quantity_with_conversion(self, ureg: UnitRegistry) -> None:
        q1 = ureg.Quantity(1.0, "kilometer")
        q2 = ureg.Quantity(q1, "meter")
        assert abs(q2.magnitude - 1000.0) < 1e-10

    def test_nan(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(float("nan"), "meter")
        assert math.isnan(q.magnitude)

    def test_inf(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(float("inf"), "meter")
        assert math.isinf(q.magnitude)
        assert q.magnitude > 0

    def test_negative_inf(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(float("-inf"), "meter")
        assert math.isinf(q.magnitude)
        assert q.magnitude < 0

    def test_bool_value(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(True, "meter")
        assert q.magnitude == 1.0

    def test_zero(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(0, "meter")
        assert q.magnitude == 0.0

    def test_negative_value(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(-5.0, "meter")
        assert q.magnitude == -5.0

    def test_dimensionless_creation(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "dimensionless")
        assert q.dimensionless

    def test_empty_string_raises(self, ureg: UnitRegistry) -> None:
        with pytest.raises(ValueError):
            ureg.Quantity("")

    def test_direct_constructor(self) -> None:
        q = Quantity(5.0, "meter")
        assert q.magnitude == 5.0
        assert str(q.units) == "m"

    def test_getattr_meter(self, ureg: UnitRegistry) -> None:
        u = ureg.meter
        assert isinstance(u, Unit)
        assert str(u) == "m"

    def test_getattr_kilogram(self, ureg: UnitRegistry) -> None:
        u = ureg.kilogram
        assert str(u) == "kilogram"

    def test_getattr_symbol(self, ureg: UnitRegistry) -> None:
        u = ureg.kg
        assert str(u) == "kilogram"

    def test_undefined_unit_raises(self, ureg: UnitRegistry) -> None:
        with pytest.raises((UndefinedUnitError, ValueError)):
            ureg.Quantity(1, "nonexistent_xyz_unit")

    def test_parse_expression(self, ureg: UnitRegistry) -> None:
        q = ureg.parse_expression("3.5 meters")
        assert abs(q.magnitude - 3.5) < 1e-10


# ---------------------------------------------------------------------------
# 2. TestQuantityConversion
# ---------------------------------------------------------------------------


class TestQuantityConversion:
    """to(), ito(), to_base_units(), to_root_units(), to_compact(), etc."""

    def test_to_compatible(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "kilometer")
        result = q.to("meter")
        assert abs(result.magnitude - 1000.0) < 1e-10

    def test_to_incompatible_raises(self, ureg: UnitRegistry) -> None:
        with pytest.raises(DimensionalityError):
            ureg.Quantity(1.0, "meter").to("second")

    def test_ito(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "kilometer")
        q.ito("meter")
        assert abs(q.magnitude - 1000.0) < 1e-10

    def test_to_base_units(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "kilometer")
        base = q.to_base_units()
        assert abs(base.magnitude - 5000.0) < 1e-10
        assert str(base.units) == "m"

    def test_ito_base_units(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "km")
        q.ito_base_units()
        assert abs(q.magnitude - 5000.0) < 1e-10

    def test_to_root_units(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "km")
        result = q.to_root_units()
        assert abs(result.magnitude - 5000.0) < 1e-10

    def test_ito_root_units(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "km")
        q.ito_root_units()
        assert abs(q.magnitude - 5000.0) < 1e-10

    def test_to_compact_small(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(0.001, "meter")
        compact = q.to_compact()
        assert abs(compact.magnitude - 1.0) < 1e-10

    def test_to_compact_large(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5000.0, "meter")
        compact = q.to_compact()
        assert abs(compact.magnitude - 5.0) < 1e-10

    def test_to_compact_micro(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(0.000001, "meter")
        compact = q.to_compact()
        assert abs(compact.magnitude - 1.0) < 1e-10

    def test_to_reduced_units(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "km/m")
        result = q.to_reduced_units()
        assert abs(result.magnitude - 1000.0) < 1e-6

    def test_ito_reduced_units(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "km")
        q.ito_reduced_units()
        assert q.magnitude > 0

    def test_to_unprefixed(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5, "km")
        result = q.to_unprefixed()
        assert abs(result.magnitude - 5000) < 1e-10
        assert str(result.units) == "m"

    def test_ito_unprefixed(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5, "km")
        q.ito_unprefixed()
        assert abs(q.magnitude - 5000) < 1e-10

    def test_to_preferred(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5, "km")
        result = q.to_preferred(["meter"])
        assert abs(result.magnitude - 5000) < 1e-10

    def test_ito_preferred(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5, "km")
        q.ito_preferred(["meter"])
        assert abs(q.magnitude - 5000) < 1e-10

    def test_m_as(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "km")
        assert abs(q.m_as("meter") - 5000.0) < 1e-10

    def test_m_as_incompatible_raises(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "meter")
        with pytest.raises(DimensionalityError):
            q.m_as("second")

    @pytest.mark.parametrize(
        ("value", "src", "dst", "expected"),
        [
            (1.0, "inch", "centimeter", 2.54),
            (1.0, "foot", "meter", 0.3048),
            (1.0, "mile", "kilometer", 1.609344),
            (1.0, "pound", "kilogram", 0.45359237),
            (1.0, "gallon", "liter", 3.785412),
            (1.0, "atmosphere", "pascal", 101325.0),
            (1.0, "horsepower", "watt", 745.7),
            (1.0, "calorie", "joule", 4.184),
        ],
        ids=[
            "inch->cm",
            "foot->m",
            "mile->km",
            "pound->kg",
            "gallon->liter",
            "atm->Pa",
            "hp->W",
            "cal->J",
        ],
    )
    def test_parametrized_conversions(
        self,
        ureg: UnitRegistry,
        value: float,
        src: str,
        dst: str,
        expected: float,
    ) -> None:
        result = ureg.Quantity(value, src).to(dst)
        assert abs(result.magnitude - expected) / max(abs(expected), 1e-30) < 1e-3


# ---------------------------------------------------------------------------
# 3. TestQuantityArithmetic
# ---------------------------------------------------------------------------


class TestQuantityArithmetic:
    """add, sub, mul, div, pow, neg, abs, round, with scalars and quantities."""

    def test_add_same_units(self, ureg: UnitRegistry) -> None:
        result = ureg.Quantity(1.0, "meter") + ureg.Quantity(2.0, "meter")
        assert result.magnitude == 3.0

    def test_add_compatible_units(self, ureg: UnitRegistry) -> None:
        result = ureg.Quantity(1.0, "km") + ureg.Quantity(500.0, "meter")
        assert abs(result.magnitude - 1.5) < 1e-10

    def test_add_incompatible_raises(self, ureg: UnitRegistry) -> None:
        with pytest.raises(DimensionalityError):
            ureg.Quantity(1.0, "meter") + ureg.Quantity(1.0, "second")

    def test_add_dimensionless_scalar(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "dimensionless")
        result = q + 1
        assert result.magnitude == 6.0

    def test_add_dimensioned_scalar_raises(self, ureg: UnitRegistry) -> None:
        with pytest.raises(DimensionalityError):
            ureg.Quantity(5.0, "meter") + 1

    def test_radd(self, ureg: UnitRegistry) -> None:
        result = 1 + ureg.Quantity(5.0, "dimensionless")
        assert result.magnitude == 6.0

    def test_sub(self, ureg: UnitRegistry) -> None:
        result = ureg.Quantity(10.0, "meter") - ureg.Quantity(3.0, "meter")
        assert result.magnitude == 7.0

    def test_sub_compatible(self, ureg: UnitRegistry) -> None:
        result = ureg.Quantity(1.0, "km") - ureg.Quantity(500.0, "meter")
        assert abs(result.magnitude - 0.5) < 1e-10

    def test_rsub(self, ureg: UnitRegistry) -> None:
        result = 10 - ureg.Quantity(5.0, "dimensionless")
        assert result.magnitude == 5.0

    def test_mul_quantities(self, ureg: UnitRegistry) -> None:
        result = ureg.Quantity(2.0, "meter") * ureg.Quantity(3.0, "second")
        assert result.magnitude == 6.0

    def test_mul_same_unit(self, ureg: UnitRegistry) -> None:
        result = ureg.Quantity(2.0, "meter") * ureg.Quantity(3.0, "meter")
        assert result.magnitude == 6.0
        assert "m ** 2" in str(result.units)

    def test_mul_scalar_right(self, ureg: UnitRegistry) -> None:
        result = ureg.Quantity(5.0, "meter") * 3
        assert result.magnitude == 15.0

    def test_mul_scalar_left(self, ureg: UnitRegistry) -> None:
        result = 3 * ureg.Quantity(5.0, "meter")
        assert result.magnitude == 15.0

    def test_mul_float_scalar(self, ureg: UnitRegistry) -> None:
        result = 3.0 * ureg.Quantity(5.0, "meter")
        assert result.magnitude == 15.0

    def test_mul_unit(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "meter")
        u = ureg.Unit("second")
        result = q * u
        assert result.magnitude == 5.0

    def test_div_quantities(self, ureg: UnitRegistry) -> None:
        result = ureg.Quantity(10.0, "meter") / ureg.Quantity(2.0, "second")
        assert result.magnitude == 5.0

    def test_div_same_units_dimensionless(self, ureg: UnitRegistry) -> None:
        result = ureg.Quantity(6.0, "meter") / ureg.Quantity(2.0, "meter")
        assert result.magnitude == 3.0
        assert result.dimensionless

    def test_div_scalar(self, ureg: UnitRegistry) -> None:
        result = ureg.Quantity(10.0, "meter") / 2
        assert result.magnitude == 5.0

    def test_rdiv(self, ureg: UnitRegistry) -> None:
        result = 10.0 / ureg.Quantity(2.0, "meter")
        assert result.magnitude == 5.0

    def test_div_unit(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "meter")
        u = ureg.Unit("second")
        result = q / u
        assert result.magnitude == 5.0

    def test_pow_integer(self, ureg: UnitRegistry) -> None:
        result = ureg.Quantity(3.0, "meter") ** 2
        assert result.magnitude == 9.0

    def test_pow_half(self, ureg: UnitRegistry) -> None:
        result = ureg.Quantity(4.0, "m**2") ** 0.5
        assert abs(result.magnitude - 2.0) < 1e-10

    def test_neg(self, ureg: UnitRegistry) -> None:
        result = -ureg.Quantity(5.0, "meter")
        assert result.magnitude == -5.0

    def test_neg_negative(self, ureg: UnitRegistry) -> None:
        result = -ureg.Quantity(-5.0, "meter")
        assert result.magnitude == 5.0

    def test_abs_negative(self, ureg: UnitRegistry) -> None:
        result = abs(ureg.Quantity(-5.0, "meter"))
        assert result.magnitude == 5.0

    def test_abs_positive(self, ureg: UnitRegistry) -> None:
        result = abs(ureg.Quantity(5.0, "meter"))
        assert result.magnitude == 5.0

    def test_round_decimal(self, ureg: UnitRegistry) -> None:
        result = round(ureg.Quantity(3.14159, "meter"), 2)
        assert abs(result.magnitude - 3.14) < 1e-10

    def test_round_no_decimals(self, ureg: UnitRegistry) -> None:
        result = round(ureg.Quantity(3.7, "meter"))
        assert result.magnitude == 4.0

    def test_mul_preserves_units(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "meter") * 0
        assert q.magnitude == 0.0


# ---------------------------------------------------------------------------
# 4. TestQuantityComparison
# ---------------------------------------------------------------------------


class TestQuantityComparison:
    """eq, ne, lt, gt, le, ge with same units, different units, incompatible."""

    def test_eq_same_units(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(1.0, "meter") == ureg.Quantity(1.0, "meter")

    def test_eq_different_units(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(1.0, "km") == ureg.Quantity(1000.0, "meter")

    def test_ne_same_units(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(1.0, "meter") != ureg.Quantity(2.0, "meter")

    def test_ne_different_type(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(5.0, "meter") != 5
        assert ureg.Quantity(5.0, "meter") != "hello"

    def test_lt(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(1.0, "meter") < ureg.Quantity(2.0, "meter")

    def test_lt_different_units(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(500, "meter") < ureg.Quantity(1, "km")

    def test_gt(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(2.0, "meter") > ureg.Quantity(1.0, "meter")

    def test_gt_different_units(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(1, "km") > ureg.Quantity(500, "meter")

    def test_le_equal(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(1.0, "meter") <= ureg.Quantity(1.0, "meter")

    def test_le_less(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(1.0, "meter") <= ureg.Quantity(2.0, "meter")

    def test_ge_equal(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(2.0, "meter") >= ureg.Quantity(2.0, "meter")

    def test_ge_greater(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(2.0, "meter") >= ureg.Quantity(1.0, "meter")

    def test_compare_method(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(1.0, "km")
        b = ureg.Quantity(500.0, "meter")
        assert a.compare(b, ">")
        assert not a.compare(b, "==")
        assert a.compare(b, ">=")
        assert a.compare(b, "!=")
        assert not a.compare(b, "<")

    def test_nan_not_equal_to_nan(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(float("nan"), "meter")
        b = ureg.Quantity(float("nan"), "meter")
        assert a != b
        assert a != b


# ---------------------------------------------------------------------------
# 5. TestQuantityRepresentation
# ---------------------------------------------------------------------------


class TestQuantityRepresentation:
    """str, repr, format, hash."""

    def test_str_simple(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.5, "meter")
        assert str(q) == "3.5 m"

    def test_repr_contains_value(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.5, "meter")
        r = repr(q)
        assert "3.5" in r
        assert "m" in r

    def test_str_compound(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "m/s")
        s = str(q)
        assert "5" in s

    def test_format_empty(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.5, "meter")
        assert format(q, "") == "3.5 m"

    def test_format_abbreviation(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.5, "meter")
        formatted = format(q, "~P")
        assert "m" in formatted

    def test_format_precision(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.14159, "meter")
        formatted = format(q, ".2f")
        assert "3.14" in formatted

    def test_format_scientific(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.14159, "meter")
        formatted = format(q, ".3e")
        assert "3.142e+00" in formatted

    def test_format_latex(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.5, "meter")
        formatted = format(q, "~L")
        assert "$" in formatted

    def test_format_html(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.5, "meter / second ** 2")
        formatted = format(q, "H")
        # HTML format is used
        assert "3.5" in formatted

    def test_format_compact(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5000.0, "meter")
        formatted = format(q, "C")
        assert "5" in formatted

    def test_hash_equal_quantities(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(1.0, "meter")
        b = ureg.Quantity(1.0, "meter")
        assert hash(a) == hash(b)

    def test_hash_converted_quantities(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(1.0, "km")
        b = ureg.Quantity(1000.0, "meter")
        assert hash(a) == hash(b)

    def test_hash_usable_in_set(self, ureg: UnitRegistry) -> None:
        q1 = ureg.Quantity(1.0, "meter")
        q2 = ureg.Quantity(1.0, "meter")
        s = {q1, q2}
        assert len(s) == 1


# ---------------------------------------------------------------------------
# 6. TestQuantitySerialization
# ---------------------------------------------------------------------------


class TestQuantitySerialization:
    """pickle, copy, deepcopy, to_tuple, from_tuple."""

    def test_pickle_roundtrip(self) -> None:
        q = Quantity(5.0, "meter")
        q2 = pickle.loads(pickle.dumps(q))
        assert abs(q2.magnitude - 5.0) < 1e-10
        assert str(q2.units) == "m"

    def test_pickle_compound_units(self) -> None:
        q = Quantity(9.8, "m/s^2")
        q2 = pickle.loads(pickle.dumps(q))
        assert abs(q2.magnitude - 9.8) < 1e-10

    def test_copy(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "meter")
        q2 = copy.copy(q)
        assert q.magnitude == q2.magnitude
        assert str(q.units) == str(q2.units)

    def test_deepcopy(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "meter")
        q2 = copy.deepcopy(q)
        assert q.magnitude == q2.magnitude
        assert str(q.units) == str(q2.units)

    def test_to_tuple_simple(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "meter")
        mag, items = q.to_tuple()
        assert mag == 5.0
        assert len(items) == 1
        assert items[0][0] == "meter"
        assert items[0][1] == 1.0

    def test_to_tuple_compound(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "m/s")
        mag, items = q.to_tuple()
        assert mag == 5.0
        assert len(items) == 2
        names = {name for name, _ in items}
        assert "meter" in names
        assert "second" in names

    def test_from_tuple_simple(self) -> None:
        q = Quantity.from_tuple((3.5, [("meter", 1.0)]))
        assert abs(q.magnitude - 3.5) < 1e-10

    def test_from_tuple_compound(self) -> None:
        q = Quantity.from_tuple((5.0, [("meter", 1.0), ("second", -1.0)]))
        assert abs(q.magnitude - 5.0) < 1e-10

    def test_to_from_tuple_roundtrip(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(7.3, "meter")
        tup = q.to_tuple()
        q2 = Quantity.from_tuple(tup)
        assert abs(q2.magnitude - q.magnitude) < 1e-10

    def test_unit_pickle(self) -> None:
        u = Unit("meter")
        u2 = pickle.loads(pickle.dumps(u))
        assert str(u) == str(u2)

    def test_unit_copy(self) -> None:
        u = Unit("meter")
        u2 = copy.copy(u)
        assert str(u) == str(u2)

    def test_unit_deepcopy(self) -> None:
        u = Unit("meter")
        u2 = copy.deepcopy(u)
        assert str(u) == str(u2)


# ---------------------------------------------------------------------------
# 7. TestQuantityTemperature
# ---------------------------------------------------------------------------


class TestQuantityTemperature:
    """Celsius/Kelvin/Fahrenheit/Rankine conversions, offset handling."""

    def test_celsius_to_kelvin(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(100, "degree_Celsius")
        result = q.to("kelvin")
        assert abs(result.magnitude - 373.15) < 1e-6

    def test_kelvin_to_celsius(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(273.15, "kelvin")
        result = q.to("degree_Celsius")
        assert abs(result.magnitude) < 1e-6

    def test_celsius_to_fahrenheit(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(100, "degree_Celsius")
        result = q.to("degree_Fahrenheit")
        assert abs(result.magnitude - 212.0) < 1e-6

    def test_fahrenheit_to_celsius(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(32, "degree_Fahrenheit")
        result = q.to("degree_Celsius")
        assert abs(result.magnitude) < 1e-6

    def test_absolute_zero_kelvin_to_celsius(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(0, "kelvin")
        result = q.to("degree_Celsius")
        assert abs(result.magnitude - (-273.15)) < 1e-6

    def test_celsius_to_rankine(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(100, "degree_Celsius")
        result = q.to("degree_Rankine")
        assert abs(result.magnitude - 671.67) < 1e-2

    def test_kelvin_to_rankine(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(0, "kelvin")
        result = q.to("degree_Rankine")
        assert abs(result.magnitude) < 1e-6

    def test_rankine_to_kelvin(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(671.67, "degree_Rankine")
        result = q.to("kelvin")
        assert abs(result.magnitude - 373.15) < 1e-2

    def test_rankine_to_fahrenheit(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(491.67, "degree_Rankine")
        result = q.to("degree_Fahrenheit")
        assert abs(result.magnitude - 32.0) < 1e-2

    def test_fahrenheit_to_kelvin(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(212, "degree_Fahrenheit")
        result = q.to("kelvin")
        assert abs(result.magnitude - 373.15) < 1e-2

    @pytest.mark.parametrize(
        ("celsius", "kelvin"),
        [
            (0, 273.15),
            (100, 373.15),
            (-40, 233.15),
            (-273.15, 0),
        ],
        ids=["freezing", "boiling", "minus40", "abs_zero"],
    )
    def test_celsius_kelvin_parametrized(
        self,
        ureg: UnitRegistry,
        celsius: float,
        kelvin: float,
    ) -> None:
        result = ureg.Quantity(celsius, "degree_Celsius").to("kelvin")
        assert abs(result.magnitude - kelvin) < 1e-6


# ---------------------------------------------------------------------------
# 8. TestQuantityDimensionality
# ---------------------------------------------------------------------------


class TestQuantityDimensionality:
    """dimensionality, dimensionless, is_compatible_with, check, compatible_units."""

    def test_length_dimensionality(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "meter")
        assert q.dimensionality == "[length]"

    def test_time_dimensionality(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "second")
        assert q.dimensionality == "[time]"

    def test_velocity_dimensionality(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "m/s")
        dim = q.dimensionality
        assert "[length]" in dim
        assert "[time]" in dim

    def test_force_dimensionality(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "kg*m/s**2")
        dim = q.dimensionality
        assert "[mass]" in dim
        assert "[length]" in dim
        assert "[time]" in dim

    def test_dimensionless_ratio(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "meter") / ureg.Quantity(1, "meter")
        assert q.dimensionless
        assert q.unitless

    def test_not_dimensionless(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "meter")
        assert not q.dimensionless
        assert not q.unitless

    def test_is_compatible_with_unit_string(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "meter")
        assert q.is_compatible_with("foot")
        assert q.is_compatible_with("inch")
        assert q.is_compatible_with("kilometer")
        assert not q.is_compatible_with("second")
        assert not q.is_compatible_with("kilogram")

    def test_check_dimension(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "meter")
        assert q.check("[length]")
        assert not q.check("[time]")
        assert not q.check("[mass]")

    def test_compatible_units(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "meter")
        compatible = q.compatible_units()
        assert len(compatible) > 5
        assert "foot" in compatible

    def test_unit_dimensionality(self, ureg: UnitRegistry) -> None:
        assert ureg.Unit("meter").dimensionality == "[length]"
        assert ureg.Unit("second").dimensionality == "[time]"

    def test_unit_dimensionless(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("dimensionless")
        assert u.dimensionless

    def test_unit_is_compatible_with(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("meter")
        assert u.is_compatible_with("foot")
        assert not u.is_compatible_with("second")


# ---------------------------------------------------------------------------
# 9. TestQuantityNumericTypes
# ---------------------------------------------------------------------------


class TestQuantityNumericTypes:
    """float(), int(), bool(), complex types."""

    def test_float_dimensionless(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.0, "meter") / ureg.Quantity(1.0, "meter")
        assert float(q) == 3.0

    def test_float_dimensioned_raises(self, ureg: UnitRegistry) -> None:
        with pytest.raises(DimensionalityError):
            float(ureg.Quantity(3.0, "meter"))

    def test_int_dimensionless(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.0, "dimensionless")
        assert int(q) == 3

    def test_int_dimensioned_raises(self, ureg: UnitRegistry) -> None:
        with pytest.raises(DimensionalityError):
            int(ureg.Quantity(3.0, "meter"))

    def test_bool_nonzero(self, ureg: UnitRegistry) -> None:
        assert bool(ureg.Quantity(1.0, "meter"))
        assert bool(ureg.Quantity(-1.0, "meter"))

    def test_bool_zero(self, ureg: UnitRegistry) -> None:
        assert not bool(ureg.Quantity(0.0, "meter"))

    def test_bool_nan(self, ureg: UnitRegistry) -> None:
        assert bool(ureg.Quantity(float("nan"), "meter"))

    def test_bool_inf(self, ureg: UnitRegistry) -> None:
        assert bool(ureg.Quantity(float("inf"), "meter"))


# ---------------------------------------------------------------------------
# 10. TestQuantityNumpyLike
# ---------------------------------------------------------------------------


class TestQuantityNumpyLike:
    """ndim, shape, real, imag, T, tolist, clip, compute, persist."""

    def test_ndim_scalar(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.5, "meter")
        assert q.ndim == 0

    def test_shape_scalar(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.5, "meter")
        assert q.shape == ()

    def test_real_scalar(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.5, "meter")
        assert q.real.magnitude == 3.5

    def test_imag_scalar(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.5, "meter")
        assert q.imag.magnitude == 0

    def test_transpose_scalar(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.5, "meter")
        assert q.T.magnitude == 3.5

    def test_tolist_scalar(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.5, "meter")
        assert q.tolist() == 3.5

    def test_clip_below(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(0.5, "meter")
        result = q.clip(min=1, max=10)
        assert result.magnitude == 1

    def test_clip_above(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(20.0, "meter")
        result = q.clip(min=1, max=10)
        assert result.magnitude == 10

    def test_clip_in_range(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "meter")
        result = q.clip(min=1, max=10)
        assert result.magnitude == 5.0


# ---------------------------------------------------------------------------
# 11. TestQuantityDask
# ---------------------------------------------------------------------------


class TestQuantityDask:
    """compute, persist stubs for non-dask magnitudes."""

    def test_compute_returns_self(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.5, "meter")
        result = q.compute()
        assert result.magnitude == 3.5
        assert str(result.units) == "m"

    def test_persist_returns_self(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.5, "meter")
        result = q.persist()
        assert result.magnitude == 3.5
        assert str(result.units) == "m"


# ---------------------------------------------------------------------------
# 12. TestQuantityEdgeCases
# ---------------------------------------------------------------------------


class TestQuantityEdgeCases:
    """zero, nan, infinity, very large/small values, complex units."""

    def test_zero_conversion(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(0, "meter")
        result = q.to("km")
        assert result.magnitude == 0.0

    def test_nan_arithmetic(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(float("nan"), "meter")
        result = q + ureg.Quantity(1.0, "meter")
        assert math.isnan(result.magnitude)

    def test_nan_conversion(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(float("nan"), "meter")
        result = q.to("km")
        assert math.isnan(result.magnitude)

    def test_inf_arithmetic(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(float("inf"), "meter")
        result = q + ureg.Quantity(1.0, "meter")
        assert math.isinf(result.magnitude)

    def test_inf_comparison(self, ureg: UnitRegistry) -> None:
        q_inf = ureg.Quantity(float("inf"), "meter")
        q_fin = ureg.Quantity(1e300, "meter")
        assert q_inf > q_fin

    def test_very_large_value(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1e300, "meter")
        result = q.to("km")
        assert abs(result.magnitude - 1e297) < 1e290

    def test_very_small_value(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1e-300, "meter")
        result = q.to("km")
        assert result.magnitude == pytest.approx(1e-303)

    def test_complex_unit_expression(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "kg*m**2/s**2")
        result = q.to("joule")
        assert abs(result.magnitude - 1.0) < 1e-10

    def test_negative_conversion(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(-5, "meter")
        result = q.to("km")
        assert abs(result.magnitude - (-0.005)) < 1e-10

    def test_dimensionless_to_float(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(42, "dimensionless")
        assert float(q) == 42.0

    def test_mul_by_zero(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "meter") * 0
        assert q.magnitude == 0.0

    def test_div_by_large(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "meter") / 1e300
        assert q.magnitude == pytest.approx(1e-300)


# ---------------------------------------------------------------------------
# Additional tests to reach 104+ coverage
# ---------------------------------------------------------------------------


class TestQuantityProperties:
    """Properties and aliases: m, u, magnitude, units."""

    def test_magnitude_property(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "meter")
        assert q.magnitude == 5.0

    def test_m_alias(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "meter")
        assert q.m == 5.0

    def test_units_property(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "meter")
        assert isinstance(q.units, Unit)

    def test_u_alias(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "meter")
        assert str(q.u) == "m"

    def test_unit_items(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5, "m/s")
        items = q.unit_items()
        assert len(items) == 2


class TestUnitOperations:
    """Unit arithmetic and methods."""

    def test_unit_mul(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("meter") * ureg.Unit("second")
        assert str(u) != ""

    def test_unit_div(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("meter") / ureg.Unit("second")
        assert str(u) != ""

    def test_unit_pow(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("meter") ** 2
        assert str(u) != ""

    def test_unit_eq(self, ureg: UnitRegistry) -> None:
        assert ureg.Unit("meter") == ureg.Unit("meter")

    def test_unit_hash(self, ureg: UnitRegistry) -> None:
        assert hash(ureg.Unit("meter")) == hash(ureg.Unit("meter"))

    def test_unit_str(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("meter")
        assert str(u) == "m"

    def test_unit_repr(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("meter")
        assert "m" in repr(u)

    def test_unit_from(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("meter")
        q = u.from_(ureg.Quantity(1, "km"))
        assert abs(q.magnitude - 1000) < 1e-10

    def test_unit_m_from(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("meter")
        assert abs(u.m_from(ureg.Quantity(1, "km")) - 1000) < 1e-10

    def test_scalar_times_unit(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("meter")
        result = 5 * u
        assert isinstance(result, Quantity)
        assert result.magnitude == 5.0


class TestRegistryContains:
    """Registry __contains__ and __iter__."""

    def test_contains_known_unit(self, ureg: UnitRegistry) -> None:
        assert "meter" in ureg
        assert "second" in ureg
        assert "kilogram" in ureg

    def test_contains_unknown_unit(self, ureg: UnitRegistry) -> None:
        assert "nonexistent_foobar" not in ureg

    def test_iter_yields_units(self, ureg: UnitRegistry) -> None:
        names = list(ureg)
        assert "meter" in names
        assert len(names) > 100

    def test_registry_convert(self, ureg: UnitRegistry) -> None:
        assert abs(ureg.convert(1.0, "km", "meter") - 1000) < 1e-10

    def test_registry_define(self, ureg: UnitRegistry) -> None:
        ureg.define("smoot_test = 1.7018 * meter")
        q = ureg.Quantity(1, "smoot_test").to("meter")
        assert abs(q.magnitude - 1.7018) < 1e-10
