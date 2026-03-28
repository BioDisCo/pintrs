"""Regression and edge-case tests - ported from pint's test_issues.py."""

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
    """Fresh unit registry for each test."""
    return UnitRegistry()


# ---------------------------------------------------------------------------
# 1. Creation edge cases
# ---------------------------------------------------------------------------


class TestCreationEdgeCases:
    """Edge cases in Quantity and Unit construction."""

    def test_zero_quantity(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(0, "meter")
        assert q.magnitude == 0.0
        assert str(q.units) == "m"

    def test_negative_quantity(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(-42.5, "second")
        assert q.magnitude == -42.5

    def test_very_large_magnitude(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1e300, "meter")
        assert q.magnitude == 1e300

    def test_very_small_magnitude(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1e-300, "meter")
        assert q.magnitude == 1e-300

    def test_nan_magnitude(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(float("nan"), "meter")
        assert math.isnan(q.magnitude)

    def test_inf_magnitude(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(float("inf"), "meter")
        assert math.isinf(q.magnitude)
        assert q.magnitude > 0

    def test_negative_inf_magnitude(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(float("-inf"), "meter")
        assert math.isinf(q.magnitude)
        assert q.magnitude < 0

    def test_quantity_from_quantity(self, ureg: UnitRegistry) -> None:
        q1 = ureg.Quantity(4.2, "meter")
        q2 = ureg.Quantity(q1)
        assert q2.magnitude == 4.2
        assert str(q2.units) == "m"

    def test_quantity_from_quantity_with_conversion(self, ureg: UnitRegistry) -> None:
        q1 = ureg.Quantity(1.0, "kilometer")
        q2 = ureg.Quantity(q1, "meter")
        assert abs(q2.magnitude - 1000.0) < 1e-10

    def test_quantity_from_string(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity("4.2*meter")
        assert abs(q.magnitude - 4.2) < 1e-10

    def test_quantity_from_string_compound(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity("9.81 meter / second ** 2")
        assert abs(q.magnitude - 9.81) < 1e-10

    def test_empty_string_raises(self, ureg: UnitRegistry) -> None:
        with pytest.raises(ValueError):
            ureg.Quantity("")

    def test_undefined_unit_raises(self, ureg: UnitRegistry) -> None:
        with pytest.raises(UndefinedUnitError):
            ureg.Quantity(1, "gibberish_unit_xyz")

    def test_dimensionless_quantity(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "dimensionless")
        assert q.dimensionless
        assert q.unitless

    def test_unit_from_string(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("meter")
        assert str(u) == "m"

    def test_unit_symbol_access(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("m")
        assert u == ureg.Unit("meter")

    def test_registry_attribute_access(self, ureg: UnitRegistry) -> None:
        u = ureg.meter
        assert isinstance(u, Unit)
        assert str(u) == "m"

    def test_registry_attribute_prefix(self, ureg: UnitRegistry) -> None:
        u = ureg.kilometer
        assert isinstance(u, Unit)
        assert str(u) == "kilometer"


# ---------------------------------------------------------------------------
# 2. Conversion edge cases
# ---------------------------------------------------------------------------


class TestConversionEdgeCases:
    """Edge cases in unit conversion."""

    def test_self_conversion_identity(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(42.0, "meter")
        result = q.to("meter")
        assert result.magnitude == 42.0

    def test_conversion_preserves_value(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "kilometer")
        result = q.to("meter").to("kilometer")
        assert abs(result.magnitude - 1.0) < 1e-10

    def test_chained_conversion(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "mile")
        result = q.to("kilometer").to("meter").to("centimeter")
        expected = 1.609344e5
        assert abs(result.magnitude - expected) < 1

    def test_compound_unit_conversion(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "kilowatt * hour")
        result = q.to("joule")
        assert abs(result.magnitude - 3.6e6) < 1

    def test_prefixed_to_base(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "millimeter")
        result = q.to("meter")
        assert abs(result.magnitude - 0.001) < 1e-12

    def test_base_to_prefixed(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1000.0, "meter")
        result = q.to("kilometer")
        assert abs(result.magnitude - 1.0) < 1e-10

    def test_incompatible_conversion_raises(self, ureg: UnitRegistry) -> None:
        with pytest.raises(DimensionalityError):
            ureg.Quantity(1, "meter").to("second")

    def test_ito_modifies_in_place(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "kilometer")
        q.ito("meter")
        assert abs(q.magnitude - 1000.0) < 1e-10

    def test_m_as_returns_float(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "kilometer")
        val = q.m_as("meter")
        assert isinstance(val, float)
        assert abs(val - 1000.0) < 1e-10

    def test_to_base_units(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "kilometer")
        base = q.to_base_units()
        assert abs(base.magnitude - 5000.0) < 1e-10
        assert str(base.units) == "m"

    def test_to_compact_small(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(0.001, "meter")
        compact = q.to_compact()
        assert abs(compact.magnitude - 1.0) < 1e-10

    def test_to_compact_large(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1500.0, "meter")
        compact = q.to_compact()
        assert abs(compact.magnitude - 1.5) < 1e-10

    def test_to_reduced_units_cancellation(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "meter * second / second")
        reduced = q.to_reduced_units()
        assert "[time]" not in reduced.dimensionality

    def test_convert_registry_method(self, ureg: UnitRegistry) -> None:
        val = ureg.convert(1.0, "kilometer", "meter")
        assert abs(val - 1000.0) < 1e-10

    def test_to_unprefixed(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "km")
        result = q.to_unprefixed()
        assert abs(result.magnitude - 5000.0) < 1e-10
        assert str(result.units) == "m"

    @pytest.mark.parametrize(
        ("src", "dst", "expected", "tol"),
        [
            ("inch", "centimeter", 2.54, 1e-6),
            ("foot", "meter", 0.3048, 1e-6),
            ("yard", "meter", 0.9144, 1e-6),
            ("mile", "kilometer", 1.609344, 1e-4),
            ("pound", "kilogram", 0.453592, 1e-3),
            ("gallon", "liter", 3.785412, 1e-3),
            ("atmosphere", "pascal", 101325.0, 1),
            ("bar", "pascal", 1e5, 1),
            ("calorie", "joule", 4.184, 1e-3),
            ("horsepower", "watt", 745.7, 1.0),
            ("electron_volt", "joule", 1.602e-19, 1e-22),
        ],
        ids=str,
    )
    def test_common_conversions(
        self,
        ureg: UnitRegistry,
        src: str,
        dst: str,
        expected: float,
        tol: float,
    ) -> None:
        result = ureg.Quantity(1, src).to(dst)
        assert abs(result.magnitude - expected) < tol


# ---------------------------------------------------------------------------
# 3. Arithmetic edge cases
# ---------------------------------------------------------------------------


class TestArithmeticEdgeCases:
    """Edge cases in quantity arithmetic."""

    def test_add_compatible_different_units(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(1.0, "kilometer")
        b = ureg.Quantity(500.0, "meter")
        result = a + b
        assert abs(result.magnitude - 1.5) < 1e-10

    def test_add_incompatible_raises(self, ureg: UnitRegistry) -> None:
        with pytest.raises(DimensionalityError):
            ureg.Quantity(1, "meter") + ureg.Quantity(1, "second")

    def test_sub_compatible_different_units(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(2.0, "kilometer")
        b = ureg.Quantity(500.0, "meter")
        result = a - b
        assert abs(result.magnitude - 1.5) < 1e-10

    def test_mul_scalar_left(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "meter")
        assert (3.0 * q).magnitude == 15.0

    def test_mul_scalar_right(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "meter")
        assert (q * 3.0).magnitude == 15.0

    def test_mul_two_quantities(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(3.0, "meter")
        b = ureg.Quantity(4.0, "second")
        result = a * b
        assert result.magnitude == 12.0

    def test_div_scalar(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(10.0, "meter")
        assert (q / 2.0).magnitude == 5.0

    def test_rdiv_scalar(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(2.0, "meter")
        result = 10.0 / q
        assert result.magnitude == 5.0

    def test_div_same_units_gives_dimensionless(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(10.0, "meter")
        b = ureg.Quantity(5.0, "meter")
        result = a / b
        assert result.dimensionless
        assert result.magnitude == 2.0

    def test_mul_then_div_cancellation(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(6.0, "meter")
        b = ureg.Quantity(3.0, "second")
        result = (a * b) / b
        assert abs(result.magnitude - 6.0) < 1e-10

    def test_power_integer(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.0, "meter")
        result = q**2
        assert result.magnitude == 9.0
        assert "[length] ** 2" in result.dimensionality

    def test_power_negative(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(2.0, "meter")
        result = q ** (-1)
        assert result.magnitude == 0.5

    def test_neg(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "meter")
        result = -q
        assert result.magnitude == -5.0
        assert str(result.units) == "m"

    def test_abs_negative(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(-5.0, "meter")
        assert abs(q).magnitude == 5.0

    def test_abs_positive(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "meter")
        assert abs(q).magnitude == 5.0

    def test_round_with_ndigits(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.14159, "meter")
        assert abs(round(q, 2).magnitude - 3.14) < 1e-10

    def test_round_no_ndigits(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.7, "meter")
        assert round(q).magnitude == 4.0

    def test_add_dimensionless_and_scalar(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.0, "meter") / ureg.Quantity(1.0, "meter")
        result = q + 1
        assert abs(result.magnitude - 4.0) < 1e-10

    def test_mul_zero(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "meter")
        result = q * 0
        assert result.magnitude == 0.0

    def test_add_zero_quantity(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(5.0, "meter")
        b = ureg.Quantity(0.0, "meter")
        assert (a + b).magnitude == 5.0

    def test_mul_quantity_by_unit(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "meter")
        u = ureg.Unit("second")
        result = q * u
        assert result.magnitude == 5.0

    def test_div_quantity_by_unit(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(10.0, "meter")
        u = ureg.Unit("second")
        result = q / u
        assert result.magnitude == 10.0


# ---------------------------------------------------------------------------
# 4. Comparison edge cases
# ---------------------------------------------------------------------------


class TestComparisonEdgeCases:
    """Edge cases in quantity comparison."""

    def test_eq_same_units(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(1, "meter") == ureg.Quantity(1, "meter")

    def test_eq_different_units(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(1, "kilometer") == ureg.Quantity(1000, "meter")

    def test_ne_different_magnitude(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(1, "meter") != ureg.Quantity(2, "meter")

    def test_lt_same_units(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(1, "meter") < ureg.Quantity(2, "meter")

    def test_lt_cross_units(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(999, "meter") < ureg.Quantity(1, "kilometer")

    def test_gt_cross_units(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(1, "kilometer") > ureg.Quantity(999, "meter")

    def test_le_equal(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(1, "meter") <= ureg.Quantity(1, "meter")

    def test_ge_equal(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(1, "meter") >= ureg.Quantity(1, "meter")

    def test_nan_not_equal_to_self(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(float("nan"), "meter")
        assert q != q

    def test_dimensionless_eq_scalar_via_float(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3, "meter") / ureg.Quantity(1, "meter")
        assert float(q) == 3.0

    def test_hash_equal_quantities(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(1, "meter")
        b = ureg.Quantity(1, "meter")
        assert hash(a) == hash(b)

    def test_compare_method(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(1, "km")
        b = ureg.Quantity(500, "meter")
        assert a.compare(b, ">")
        assert a.compare(b, ">=")
        assert a.compare(b, "!=")
        assert not a.compare(b, "==")
        assert not a.compare(b, "<")
        assert not a.compare(b, "<=")


# ---------------------------------------------------------------------------
# 5. Temperature edge cases
# ---------------------------------------------------------------------------


class TestTemperatureEdgeCases:
    """Edge cases with offset (temperature) units."""

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

    def test_absolute_zero_kelvin_to_celsius(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(0, "kelvin")
        result = q.to("degree_Celsius")
        assert abs(result.magnitude - (-273.15)) < 1e-6

    def test_absolute_zero_kelvin_to_fahrenheit(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(0, "kelvin")
        result = q.to("degree_Fahrenheit")
        assert abs(result.magnitude - (-459.67)) < 1e-2

    def test_boiling_fahrenheit_to_kelvin(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(212, "degree_Fahrenheit")
        result = q.to("kelvin")
        assert abs(result.magnitude - 373.15) < 1e-2

    def test_rankine_to_kelvin(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(671.67, "degree_Rankine")
        result = q.to("kelvin")
        assert abs(result.magnitude - 373.15) < 1e-2

    def test_kelvin_to_rankine(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(373.15, "kelvin")
        result = q.to("degree_Rankine")
        assert abs(result.magnitude - 671.67) < 1e-2

    def test_celsius_round_trip(self, ureg: UnitRegistry) -> None:
        original = 37.0
        q = ureg.Quantity(original, "degree_Celsius")
        result = q.to("kelvin").to("degree_Celsius")
        assert abs(result.magnitude - original) < 1e-10

    def test_fahrenheit_round_trip(self, ureg: UnitRegistry) -> None:
        original = 98.6
        q = ureg.Quantity(original, "degree_Fahrenheit")
        result = q.to("kelvin").to("degree_Fahrenheit")
        assert abs(result.magnitude - original) < 1e-10

    def test_convert_registry_celsius_to_kelvin(self, ureg: UnitRegistry) -> None:
        val = ureg.convert(100, "degree_Celsius", "kelvin")
        assert abs(val - 373.15) < 1e-6


# ---------------------------------------------------------------------------
# 6. String parsing edge cases
# ---------------------------------------------------------------------------


class TestStringParsingEdgeCases:
    """Edge cases in unit expression parsing."""

    def test_parse_simple_unit(self, ureg: UnitRegistry) -> None:
        q = ureg.parse_expression("1 meter")
        assert q.magnitude == 1.0

    def test_parse_symbol(self, ureg: UnitRegistry) -> None:
        q = ureg.parse_expression("1 m")
        assert q.magnitude == 1.0

    def test_parse_plural(self, ureg: UnitRegistry) -> None:
        q = ureg.parse_expression("5 meters")
        assert q.magnitude == 5.0

    def test_parse_compound_slash(self, ureg: UnitRegistry) -> None:
        q = ureg.parse_expression("9.81 m/s^2")
        assert abs(q.magnitude - 9.81) < 1e-10

    def test_parse_compound_slash_spaced(self, ureg: UnitRegistry) -> None:
        q = ureg.parse_expression("9.81 m / s ** 2")
        assert abs(q.magnitude - 9.81) < 1e-10

    def test_parse_product_star(self, ureg: UnitRegistry) -> None:
        q = ureg.parse_expression("1 m*s")
        assert q.magnitude == 1.0

    def test_parse_product_spaced(self, ureg: UnitRegistry) -> None:
        q = ureg.parse_expression("1 m * s")
        assert q.magnitude == 1.0

    def test_parse_power_double_star(self, ureg: UnitRegistry) -> None:
        q = ureg.parse_expression("1 m**2")
        assert q.magnitude == 1.0

    def test_parse_power_caret(self, ureg: UnitRegistry) -> None:
        q = ureg.parse_expression("1 m^2")
        assert q.magnitude == 1.0

    def test_parse_scientific_notation(self, ureg: UnitRegistry) -> None:
        q = ureg.parse_expression("1e-3 kg")
        assert abs(q.magnitude - 0.001) < 1e-12

    def test_parse_expression_pure_number(self, ureg: UnitRegistry) -> None:
        q = ureg.parse_expression("6.022e23")
        assert q.dimensionless

    def test_parse_units_method(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("m/s")
        assert str(u) != ""

    def test_parse_complex_compound(self, ureg: UnitRegistry) -> None:
        q = ureg.parse_expression("1 kg * m / s ^ 2")
        assert q.magnitude == 1.0
        assert q.is_compatible_with("newton")


# ---------------------------------------------------------------------------
# 7. Dimensionality edge cases
# ---------------------------------------------------------------------------


class TestDimensionalityEdgeCases:
    """Edge cases involving dimensionality checks."""

    def test_length_dimensionality(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(1, "meter").dimensionality == "[length]"

    def test_time_dimensionality(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(1, "second").dimensionality == "[time]"

    def test_mass_dimensionality(self, ureg: UnitRegistry) -> None:
        assert "[mass]" in ureg.Quantity(1, "kilogram").dimensionality

    def test_velocity_dimensionality(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "m/s")
        dim = q.dimensionality
        assert "[length]" in dim
        assert "[time]" in dim

    def test_derived_force_dimensionality(self, ureg: UnitRegistry) -> None:
        dim = ureg.get_dimensionality("newton")
        assert "[mass]" in dim
        assert "[length]" in dim
        assert "[time]" in dim

    def test_derived_energy_dimensionality(self, ureg: UnitRegistry) -> None:
        dim = ureg.get_dimensionality("joule")
        assert "[mass]" in dim
        assert "[length]" in dim
        assert "[time]" in dim

    def test_derived_power_dimensionality(self, ureg: UnitRegistry) -> None:
        dim = ureg.get_dimensionality("watt")
        assert "[mass]" in dim
        assert "[length]" in dim
        assert "[time]" in dim

    def test_dimensionless_from_division(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(10, "meter") / ureg.Quantity(5, "meter")
        assert q.dimensionless
        assert q.unitless

    def test_area_dimensionality(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "meter ** 2")
        assert "[length] ** 2" in q.dimensionality

    def test_volume_dimensionality(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "meter ** 3")
        assert "[length] ** 3" in q.dimensionality

    def test_is_compatible_with_same_dimension(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(1, "meter").is_compatible_with("foot")
        assert ureg.Quantity(1, "meter").is_compatible_with("kilometer")
        assert ureg.Quantity(1, "meter").is_compatible_with("inch")

    def test_is_compatible_with_different_dimension(self, ureg: UnitRegistry) -> None:
        assert not ureg.Quantity(1, "meter").is_compatible_with("second")
        assert not ureg.Quantity(1, "meter").is_compatible_with("kilogram")

    def test_check_dimension(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "meter")
        assert q.check("[length]")
        assert not q.check("[time]")
        assert not q.check("[mass]")

    def test_float_dimensionless_ok(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3, "meter") / ureg.Quantity(1, "meter")
        assert float(q) == 3.0

    def test_float_dimensioned_raises(self, ureg: UnitRegistry) -> None:
        with pytest.raises(DimensionalityError):
            float(ureg.Quantity(3, "meter"))

    def test_int_dimensionless(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.7, "meter") / ureg.Quantity(1, "meter")
        assert int(q) == 3

    def test_bool_nonzero(self, ureg: UnitRegistry) -> None:
        assert bool(ureg.Quantity(1, "meter"))

    def test_bool_zero(self, ureg: UnitRegistry) -> None:
        assert not bool(ureg.Quantity(0, "meter"))

    def test_get_dimensionality_registry(self, ureg: UnitRegistry) -> None:
        assert ureg.get_dimensionality("meter") == "[length]"

    def test_get_compatible_units(self, ureg: UnitRegistry) -> None:
        units = ureg.get_compatible_units("meter")
        assert len(units) > 5


# ---------------------------------------------------------------------------
# 8. Pickle edge cases
# ---------------------------------------------------------------------------


class TestPickleEdgeCases:
    """Edge cases in serialization."""

    @pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
    def test_pickle_simple_quantity(self, protocol: int) -> None:
        q = Quantity(5.0, "meter")
        q2 = pickle.loads(pickle.dumps(q, protocol))
        assert abs(q2.magnitude - 5.0) < 1e-10
        assert str(q2.units) == "m"

    @pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
    def test_pickle_compound_quantity(self, protocol: int) -> None:
        q = Quantity(9.81, "m/s^2")
        q2 = pickle.loads(pickle.dumps(q, protocol))
        assert abs(q2.magnitude - 9.81) < 1e-10

    @pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
    def test_pickle_dimensionless(self, protocol: int) -> None:
        q = Quantity(42.0, "dimensionless")
        q2 = pickle.loads(pickle.dumps(q, protocol))
        assert q2.dimensionless
        assert q2.magnitude == 42.0

    @pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
    def test_pickle_zero_quantity(self, protocol: int) -> None:
        q = Quantity(0.0, "meter")
        q2 = pickle.loads(pickle.dumps(q, protocol))
        assert q2.magnitude == 0.0

    @pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
    def test_pickle_negative_quantity(self, protocol: int) -> None:
        q = Quantity(-7.5, "kilogram")
        q2 = pickle.loads(pickle.dumps(q, protocol))
        assert q2.magnitude == -7.5

    @pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
    def test_pickle_unit(self, protocol: int) -> None:
        u = Unit("meter")
        u2 = pickle.loads(pickle.dumps(u, protocol))
        assert str(u) == str(u2)

    @pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
    def test_pickle_compound_unit(self, protocol: int) -> None:
        u = Unit("kg*m/s^2")
        u2 = pickle.loads(pickle.dumps(u, protocol))
        assert u == u2

    def test_copy_quantity(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(42, "meter")
        q2 = copy.copy(q)
        assert q.magnitude == q2.magnitude
        assert str(q.units) == str(q2.units)

    def test_deepcopy_quantity(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(42, "meter")
        q2 = copy.deepcopy(q)
        assert q.magnitude == q2.magnitude
        assert str(q.units) == str(q2.units)

    def test_copy_unit(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("meter")
        u2 = copy.copy(u)
        assert u == u2

    def test_deepcopy_unit(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("meter")
        u2 = copy.deepcopy(u)
        assert u == u2


# ---------------------------------------------------------------------------
# 9. Custom units
# ---------------------------------------------------------------------------


class TestCustomUnits:
    """Custom unit definitions and conversions."""

    def test_define_simple(self) -> None:
        ureg = UnitRegistry()
        ureg.define("smoot = 1.7018 * meter")
        q = ureg.Quantity(1, "smoot").to("meter")
        assert abs(q.magnitude - 1.7018) < 1e-10

    def test_define_with_alias(self) -> None:
        ureg = UnitRegistry()
        ureg.define("cubit = 0.4572 * meter = cbt")
        q = ureg.Quantity(1, "cbt").to("meter")
        assert abs(q.magnitude - 0.4572) < 1e-10

    def test_define_derived_from_custom(self) -> None:
        ureg = UnitRegistry()
        ureg.define("smoot = 1.7018 * meter")
        ureg.define("half_smoot = 0.5 * smoot")
        q = ureg.Quantity(1, "half_smoot").to("meter")
        assert abs(q.magnitude - 0.8509) < 1e-10

    def test_define_time_based(self) -> None:
        ureg = UnitRegistry()
        ureg.define("dog_year = 7 * year")
        q = ureg.Quantity(3, "dog_year").to("year")
        assert abs(q.magnitude - 21.0) < 1e-10

    def test_define_does_not_affect_other_registries(self) -> None:
        ureg1 = UnitRegistry()
        _ureg2 = UnitRegistry()
        ureg1.define("xyzunit = 42 * meter")
        assert "xyzunit" in ureg1
        # New registries should not see definitions from other instances
        ureg3 = UnitRegistry()
        assert "xyzunit" not in ureg3

    def test_custom_unit_conversion_chain(self) -> None:
        ureg = UnitRegistry()
        ureg.define("foo = 2 * meter")
        q = ureg.Quantity(1, "foo").to("centimeter")
        assert abs(q.magnitude - 200.0) < 1e-10

    def test_custom_unit_in_compound(self) -> None:
        ureg = UnitRegistry()
        ureg.define("smoot = 1.7018 * meter")
        q = ureg.Quantity(10, "smoot / second")
        result = q.to("meter / second")
        assert abs(result.magnitude - 17.018) < 1e-10

    def test_load_definitions_string(self) -> None:
        ureg = UnitRegistry()
        ureg.load_definitions("test_load_unit = 99 * meter")
        assert "test_load_unit" in ureg


# ---------------------------------------------------------------------------
# 10. Physical constants
# ---------------------------------------------------------------------------


class TestPhysicalConstants:
    """Physical constants defined in the registry."""

    def test_speed_of_light(self, ureg: UnitRegistry) -> None:
        c = ureg.Quantity(1, "speed_of_light").to("m/s")
        assert abs(c.magnitude - 299792458) < 1

    def test_boltzmann_constant(self, ureg: UnitRegistry) -> None:
        k = ureg.Quantity(1, "boltzmann_constant").to("joule/kelvin")
        assert abs(k.magnitude - 1.380649e-23) < 1e-28

    def test_planck_constant(self, ureg: UnitRegistry) -> None:
        h = ureg.Quantity(1, "planck_constant").to_base_units()
        assert h.magnitude > 0
        assert h.magnitude < 1e-30

    def test_elementary_charge(self, ureg: UnitRegistry) -> None:
        e = ureg.Quantity(1, "elementary_charge").to("coulomb")
        assert abs(e.magnitude - 1.602176634e-19) < 1e-25

    def test_gravitational_constant(self, ureg: UnitRegistry) -> None:
        g = ureg.Quantity(1, "gravitational_constant").to_base_units()
        assert g.magnitude > 0
        assert g.magnitude < 1e-7

    def test_avogadro_constant(self, ureg: UnitRegistry) -> None:
        na = ureg.Quantity(1, "avogadro_constant").to_base_units()
        assert na.magnitude > 6e23
        assert na.magnitude < 7e23

    def test_electron_volt_to_joule(self, ureg: UnitRegistry) -> None:
        ev = ureg.Quantity(1, "electron_volt").to("joule")
        assert abs(ev.magnitude - 1.602e-19) < 1e-22


# ---------------------------------------------------------------------------
# 11. Unit systems
# ---------------------------------------------------------------------------


class TestUnitSystems:
    """SI, CGS, and imperial conversion tests."""

    def test_newton_to_dyne(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "newton").to("dyne")
        assert abs(q.magnitude - 1e5) < 1

    def test_joule_to_erg(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "joule").to("erg")
        assert abs(q.magnitude - 1e7) < 1

    def test_pascal_to_barye(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "pascal").to("barye")
        assert abs(q.magnitude - 10.0) < 0.1

    def test_foot_to_meter(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "foot").to("meter")
        assert abs(q.magnitude - 0.3048) < 1e-6

    def test_pound_to_kilogram(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "pound").to("kilogram")
        assert abs(q.magnitude - 0.45359237) < 1e-6

    def test_psi_to_pascal(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "psi").to("pascal")
        assert abs(q.magnitude - 6894.757) < 1

    def test_btu_to_joule(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "BTU").to("joule")
        assert abs(q.magnitude - 1055.056) < 1

    def test_acre_to_square_meter(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "acre").to("meter ** 2")
        assert abs(q.magnitude - 4046.86) < 1

    def test_liter_to_cubic_meter(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "liter").to("meter ** 3")
        assert abs(q.magnitude - 0.001) < 1e-6

    def test_gallon_to_liter(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "gallon").to("liter")
        assert abs(q.magnitude - 3.785412) < 1e-3


# ---------------------------------------------------------------------------
# 12. Compound conversions
# ---------------------------------------------------------------------------


class TestCompoundConversions:
    """Multi-step and derived quantity conversions."""

    def test_force_kg_m_s2_to_newton(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "kilogram * meter / second ** 2")
        result = q.to("newton")
        assert abs(result.magnitude - 1.0) < 1e-10

    def test_energy_kg_m2_s2_to_joule(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "kilogram * meter ** 2 / second ** 2")
        result = q.to("joule")
        assert abs(result.magnitude - 1.0) < 1e-10

    def test_pressure_kg_m_s2_to_pascal(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "kilogram / meter / second ** 2")
        result = q.to("pascal")
        assert abs(result.magnitude - 1.0) < 1e-10

    def test_power_kg_m2_s3_to_watt(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "kilogram * meter ** 2 / second ** 3")
        result = q.to("watt")
        assert abs(result.magnitude - 1.0) < 1e-10

    def test_kwh_to_joule(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "kilowatt * hour")
        result = q.to("joule")
        assert abs(result.magnitude - 3.6e6) < 1

    def test_velocity_kmh_to_ms(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(100, "km/h")
        result = q.to("m/s")
        assert abs(result.magnitude - 27.7778) < 0.001

    def test_velocity_mph_to_ms(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(60, "mile / hour")
        result = q.to("m/s")
        assert abs(result.magnitude - 26.8224) < 0.001

    def test_acceleration_to_base(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "kilometer / hour ** 2")
        result = q.to_base_units()
        assert result.magnitude > 0

    def test_density_conversion(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1000, "kilogram / meter ** 3")
        result = q.to("gram / liter")
        assert abs(result.magnitude - 1000.0) < 1

    def test_angular_velocity(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "revolution / minute")
        result = q.to("radian / second")
        expected = 2 * math.pi / 60
        assert abs(result.magnitude - expected) < 1e-6


# ---------------------------------------------------------------------------
# 13. Unit operations
# ---------------------------------------------------------------------------


class TestUnitOperations:
    """Tests for Unit object operations."""

    def test_unit_equality(self, ureg: UnitRegistry) -> None:
        assert ureg.Unit("meter") == ureg.Unit("meter")
        assert ureg.Unit("meter") == ureg.Unit("m")

    def test_unit_inequality(self, ureg: UnitRegistry) -> None:
        assert ureg.Unit("meter") != ureg.Unit("second")

    def test_unit_mul(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("meter") * ureg.Unit("second")
        assert str(u) != ""

    def test_unit_div(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("meter") / ureg.Unit("second")
        assert str(u) != ""

    def test_unit_pow(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("meter") ** 2
        assert "[length] ** 2" in u.dimensionality

    def test_unit_dimensionality(self, ureg: UnitRegistry) -> None:
        assert ureg.Unit("meter").dimensionality == "[length]"

    def test_unit_dimensionless(self, ureg: UnitRegistry) -> None:
        assert ureg.Unit("dimensionless").dimensionless
        assert not ureg.Unit("meter").dimensionless

    def test_unit_is_compatible_with(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("meter")
        assert u.is_compatible_with("foot")
        assert not u.is_compatible_with("second")

    def test_unit_from_(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("meter")
        q = u.from_(ureg.Quantity(1, "km"))
        assert abs(q.magnitude - 1000) < 1e-10

    def test_unit_m_from(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("meter")
        val = u.m_from(ureg.Quantity(1, "km"))
        assert abs(val - 1000) < 1e-10

    def test_unit_mul_scalar_gives_quantity(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("meter")
        result = 5 * u
        assert isinstance(result, Quantity)
        assert result.magnitude == 5.0

    def test_unit_compatible_units(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("meter")
        compat = u.compatible_units()
        assert len(compat) > 5


# ---------------------------------------------------------------------------
# 14. Registry operations
# ---------------------------------------------------------------------------


class TestRegistryOperations:
    """Tests for UnitRegistry methods and properties."""

    def test_contains_known_unit(self, ureg: UnitRegistry) -> None:
        assert "meter" in ureg
        assert "second" in ureg
        assert "kilogram" in ureg

    def test_not_contains_unknown(self, ureg: UnitRegistry) -> None:
        assert "xyzfoo" not in ureg

    def test_contains_prefixed(self, ureg: UnitRegistry) -> None:
        assert "kilowatt" in ureg
        assert "millimeter" in ureg

    def test_iter_returns_many_units(self, ureg: UnitRegistry) -> None:
        names = list(ureg)
        assert len(names) > 100

    def test_get_name(self, ureg: UnitRegistry) -> None:
        assert ureg.get_name("m") == "meter"
        assert ureg.get_name("s") == "second"

    def test_get_symbol(self, ureg: UnitRegistry) -> None:
        assert ureg.get_symbol("meter") == "m"
        assert ureg.get_symbol("second") == "s"

    def test_is_compatible_with(self, ureg: UnitRegistry) -> None:
        assert ureg.is_compatible_with("meter", "foot")
        assert not ureg.is_compatible_with("meter", "second")

    def test_get_base_units(self, ureg: UnitRegistry) -> None:
        factor, unit = ureg.get_base_units("kilometer")
        assert abs(factor - 1000) < 1e-10
        assert str(unit) == "m"

    def test_get_dimensionality(self, ureg: UnitRegistry) -> None:
        assert ureg.get_dimensionality("meter") == "[length]"
        assert "[time]" in ureg.get_dimensionality("hertz")

    def test_parse_unit_name(self, ureg: UnitRegistry) -> None:
        result = ureg.parse_unit_name("kilometer")
        assert len(result) > 0
        assert result[0][0] == "kilo"
        assert result[0][1] == "meter"

    def test_default_format(self, ureg: UnitRegistry) -> None:
        assert ureg.default_format == "D"

    def test_auto_reduce_dimensions(self, ureg: UnitRegistry) -> None:
        assert ureg.auto_reduce_dimensions is False


# ---------------------------------------------------------------------------
# 15. Timedelta conversion
# ---------------------------------------------------------------------------


class TestTimedelta:
    """Tests for to_timedelta()."""

    @pytest.mark.parametrize(
        ("value", "unit", "expected_seconds"),
        [
            (1, "second", 1.0),
            (60, "second", 60.0),
            (1, "minute", 60.0),
            (1.5, "minute", 90.0),
            (1, "hour", 3600.0),
            (1, "day", 86400.0),
            (1, "week", 604800.0),
            (1, "millisecond", 0.001),
        ],
    )
    def test_to_timedelta(
        self,
        ureg: UnitRegistry,
        value: float,
        unit: str,
        expected_seconds: float,
    ) -> None:
        import datetime

        q = ureg.Quantity(value, unit)
        td = q.to_timedelta()
        assert isinstance(td, datetime.timedelta)
        assert abs(td.total_seconds() - expected_seconds) < 1e-6

    def test_non_time_to_timedelta_raises(self, ureg: UnitRegistry) -> None:
        with pytest.raises(DimensionalityError):
            ureg.Quantity(1, "meter").to_timedelta()


# ---------------------------------------------------------------------------
# 16. Formatting
# ---------------------------------------------------------------------------


class TestFormatting:
    """Tests for format specifiers."""

    def test_str(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.5, "meter")
        assert str(q) == "3.5 m"

    def test_repr_contains_value(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.5, "meter")
        r = repr(q)
        assert "3.5" in r
        assert "m" in r

    def test_format_empty(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.5, "meter")
        assert format(q, "") == "3.5 m"

    def test_format_pretty(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(9.8, "m/s^2")
        result = format(q, "~P")
        assert "m" in result
        assert "s" in result

    def test_format_compact(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1500, "meter")
        result = format(q, "~C")
        assert "1.5" in result

    def test_format_with_precision(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.14159, "meter")
        result = format(q, ".2f~P")
        assert "3.14" in result

    def test_unit_format_str(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("meter")
        assert str(u) == "m"

    def test_unit_repr(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("meter")
        r = repr(u)
        assert "m" in r


# ---------------------------------------------------------------------------
# 17. Tuple round-trip
# ---------------------------------------------------------------------------


class TestTupleRoundTrip:
    """Tests for to_tuple / from_tuple."""

    def test_simple_roundtrip(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.14, "meter")
        tup = q.to_tuple()
        q2 = Quantity.from_tuple(tup)
        assert abs(q.magnitude - q2.magnitude) < 1e-10

    def test_compound_roundtrip(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(9.81, "meter/second**2")
        tup = q.to_tuple()
        q2 = Quantity.from_tuple(tup)
        assert abs(q.magnitude - q2.magnitude) < 1e-10

    def test_to_tuple_structure(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "meter")
        mag, items = q.to_tuple()
        assert mag == 5.0
        assert isinstance(items, list)
        assert len(items) >= 1
        assert items[0][0] == "meter"
        assert items[0][1] == 1.0

    def test_from_tuple_compound(self) -> None:
        tup = (9.81, [("meter", 1.0), ("second", -2.0)])
        q = Quantity.from_tuple(tup)
        assert abs(q.magnitude - 9.81) < 1e-10


# ---------------------------------------------------------------------------
# 18. Prefix coverage
# ---------------------------------------------------------------------------


class TestPrefixes:
    """Ensure common SI prefixes work correctly."""

    @pytest.mark.parametrize(
        ("prefixed", "base", "expected_factor"),
        [
            ("kilometer", "meter", 1e3),
            ("megawatt", "watt", 1e6),
            ("gigahertz", "hertz", 1e9),
            ("millimeter", "meter", 1e-3),
            ("microsecond", "second", 1e-6),
            ("nanosecond", "second", 1e-9),
            ("centimeter", "meter", 1e-2),
            ("milliampere", "ampere", 1e-3),
        ],
    )
    def test_prefix_factor(
        self,
        ureg: UnitRegistry,
        prefixed: str,
        base: str,
        expected_factor: float,
    ) -> None:
        q = ureg.Quantity(1, prefixed).to(base)
        assert abs(q.magnitude - expected_factor) / expected_factor < 1e-10


# ---------------------------------------------------------------------------
# 19. Misc regressions
# ---------------------------------------------------------------------------


class TestMiscRegressions:
    """Miscellaneous regression tests."""

    def test_quantity_magnitude_alias(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "meter")
        assert q.m == q.magnitude

    def test_quantity_units_alias(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "meter")
        assert str(q.u) == str(q.units)

    def test_unit_items(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5, "m/s")
        items = q.unit_items()
        assert len(items) == 2

    def test_quantity_in_set(self, ureg: UnitRegistry) -> None:
        q1 = ureg.Quantity(1, "meter")
        q2 = ureg.Quantity(1, "meter")
        s = {q1, q2}
        assert len(s) == 1

    def test_multiple_registries_independent(self) -> None:
        ureg1 = UnitRegistry()
        ureg2 = UnitRegistry()
        q1 = ureg1.Quantity(1, "meter")
        q2 = ureg2.Quantity(1, "meter")
        assert q1.magnitude == q2.magnitude

    def test_ito_base_units(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "kilometer")
        q.ito_base_units()
        assert abs(q.magnitude - 5000.0) < 1e-10

    def test_to_root_units(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "kilometer")
        root = q.to_root_units()
        assert abs(root.magnitude - 5000.0) < 1e-10

    def test_ito_root_units(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "kilometer")
        q.ito_root_units()
        assert abs(q.magnitude - 5000.0) < 1e-10

    def test_to_preferred(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5, "km")
        result = q.to_preferred(["meter"])
        assert abs(result.magnitude - 5000) < 1e-10

    def test_ito_preferred(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5, "km")
        q.ito_preferred(["meter"])
        assert abs(q.magnitude - 5000) < 1e-10

    def test_ito_reduced_units(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5, "km")
        q.ito_reduced_units()
        assert q.magnitude > 0

    def test_ndim_scalar(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5, "meter")
        assert q.ndim == 0

    def test_shape_scalar(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5, "meter")
        assert q.shape == ()

    def test_plus_minus(self, ureg: UnitRegistry) -> None:
        from pintrs import Measurement

        q = ureg.Quantity(5.0, "meter")
        m = q.plus_minus(0.1)
        assert isinstance(m, Measurement)
        assert m.magnitude == 5.0
