from __future__ import annotations

import pytest
from pintrs import Unit, UnitRegistry


class TestQuantityCreation:
    def test_from_value_and_units(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.5, "meter")
        assert q.magnitude == 3.5
        assert str(q.units) == "meter"

    def test_from_string(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity("9.8 m/s^2")
        assert abs(q.magnitude - 9.8) < 1e-10

    def test_parse_expression(self, ureg: UnitRegistry) -> None:
        q = ureg.parse_expression("3.5 meters")
        assert abs(q.magnitude - 3.5) < 1e-10

    def test_attribute_access(self, ureg: UnitRegistry) -> None:
        u = ureg.meter
        assert isinstance(u, Unit)
        assert str(u) == "meter"

    def test_symbol_access(self, ureg: UnitRegistry) -> None:
        u = ureg.kg
        assert isinstance(u, Unit)
        assert str(u) == "kilogram"

    def test_dimensionless(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "dimensionless")
        assert q.dimensionless

    def test_empty_string_raises(self, ureg: UnitRegistry) -> None:
        with pytest.raises(ValueError):
            ureg.Quantity("")


class TestQuantityConversion:
    def test_to(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "kilometer")
        result = q.to("meter")
        assert abs(result.magnitude - 1000.0) < 1e-10

    def test_ito(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "kilometer")
        q.ito("meter")
        assert abs(q.magnitude - 1000.0) < 1e-10

    def test_m_as(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "kilometer")
        assert abs(q.m_as("meter") - 1000.0) < 1e-10

    def test_to_base_units(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "kilometer")
        base = q.to_base_units()
        assert abs(base.magnitude - 1000.0) < 1e-10
        assert str(base.units) == "meter"

    def test_incompatible_raises(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "meter")
        with pytest.raises(ValueError):
            q.to("second")

    def test_prefixed_units(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "millimeter")
        result = q.to("meter")
        assert abs(result.magnitude - 0.001) < 1e-10

    def test_centimeter_to_inch(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(2.54, "centimeter")
        result = q.to("inch")
        assert abs(result.magnitude - 1.0) < 1e-6


class TestTemperature:
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


class TestArithmetic:
    def test_add_same_units(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(10, "meter")
        b = ureg.Quantity(5, "meter")
        result = a + b
        assert result.magnitude == 15

    def test_add_compatible_units(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(1, "kilometer")
        b = ureg.Quantity(500, "meter")
        result = a + b
        assert abs(result.magnitude - 1.5) < 1e-10

    def test_add_incompatible_raises(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(1, "meter")
        b = ureg.Quantity(1, "second")
        with pytest.raises(ValueError):
            a + b

    def test_sub(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(10, "meter")
        b = ureg.Quantity(3, "meter")
        result = a - b
        assert result.magnitude == 7

    def test_mul_quantities(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(10, "meter")
        b = ureg.Quantity(3, "second")
        result = a * b
        assert result.magnitude == 30

    def test_mul_scalar(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5, "meter")
        assert (q * 3).magnitude == 15
        assert (3 * q).magnitude == 15

    def test_div_quantities(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(10, "meter")
        b = ureg.Quantity(2, "second")
        result = a / b
        assert result.magnitude == 5.0

    def test_div_scalar(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(10, "meter")
        assert (q / 2).magnitude == 5.0

    def test_rdiv(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(2, "meter")
        result = 10.0 / q
        assert result.magnitude == 5.0

    def test_pow(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3, "meter")
        result = q**2
        assert result.magnitude == 9

    def test_neg(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5, "meter")
        assert (-q).magnitude == -5

    def test_abs(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(-5, "meter")
        assert abs(q).magnitude == 5


class TestComparison:
    def test_eq_same(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(10, "meter")
        b = ureg.Quantity(10, "meter")
        assert a == b

    def test_eq_different_units(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(1, "kilometer")
        b = ureg.Quantity(1000, "meter")
        assert a == b

    def test_neq(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(10, "meter")
        b = ureg.Quantity(5, "meter")
        assert a != b

    def test_lt(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(5, "meter") < ureg.Quantity(10, "meter")

    def test_gt(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(10, "meter") > ureg.Quantity(5, "meter")

    def test_le(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(5, "meter") <= ureg.Quantity(5, "meter")
        assert ureg.Quantity(5, "meter") <= ureg.Quantity(10, "meter")

    def test_ge(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(10, "meter") >= ureg.Quantity(10, "meter")
        assert ureg.Quantity(10, "meter") >= ureg.Quantity(5, "meter")


class TestDimensionality:
    def test_length(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "meter")
        assert q.dimensionality == "[length]"
        assert not q.dimensionless

    def test_velocity(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "m/s")
        assert "[length]" in q.dimensionality
        assert "[time]" in q.dimensionality

    def test_dimensionless(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "meter") / ureg.Quantity(1, "meter")
        assert q.dimensionless

    def test_is_compatible_with(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "meter")
        assert q.is_compatible_with("foot")
        assert q.is_compatible_with("kilometer")
        assert not q.is_compatible_with("second")
        assert not q.is_compatible_with("kilogram")

    def test_float_dimensionless(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3, "meter") / ureg.Quantity(1, "meter")
        assert float(q) == 3.0

    def test_float_non_dimensionless_raises(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3, "meter")
        with pytest.raises(ValueError):
            float(q)


class TestRepr:
    def test_repr(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.5, "meter")
        assert "3.5" in repr(q)
        assert "m" in repr(q)

    def test_str(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.5, "meter")
        assert str(q) == "3.5 meter"

    def test_hash(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(1, "meter")
        b = ureg.Quantity(1, "meter")
        assert hash(a) == hash(b)


class TestVariousUnits:
    def test_atmosphere_to_pascal(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "atmosphere")
        result = q.to("pascal")
        assert abs(result.magnitude - 101325) < 1

    def test_mile_to_km(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "mile")
        result = q.to("kilometer")
        assert abs(result.magnitude - 1.609344) < 1e-4

    def test_gallon_to_liter(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "gallon")
        result = q.to("liter")
        assert abs(result.magnitude - 3.785412) < 1e-3

    def test_horsepower_to_watt(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "horsepower")
        result = q.to("watt")
        assert abs(result.magnitude - 745.7) < 1.0

    def test_electron_volt(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "electron_volt")
        result = q.to("joule")
        assert abs(result.magnitude - 1.602e-19) < 1e-22

    def test_speed_of_light(self, ureg: UnitRegistry) -> None:
        c = ureg.Quantity(1, "speed_of_light")
        result = c.to("m/s")
        assert abs(result.magnitude - 299792458) < 1
