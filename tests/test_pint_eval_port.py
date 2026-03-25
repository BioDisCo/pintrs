"""Expression parsing/evaluation tests - ported from pint's test_pint_eval.py.

Tests expression parsing through ureg.parse_expression() and Quantity()
string parsing rather than pint's internal build_eval_tree/tokenizer.
"""

from __future__ import annotations

import pytest
from pintrs import UnitRegistry


@pytest.fixture
def ureg() -> UnitRegistry:
    return UnitRegistry()


class TestParseSimpleExpressions:
    """Test parsing numeric expressions."""

    def test_parse_integer(self, ureg: UnitRegistry) -> None:
        q = ureg.parse_expression("3 dimensionless")
        assert abs(q.magnitude - 3) < 1e-10

    def test_parse_float(self, ureg: UnitRegistry) -> None:
        q = ureg.parse_expression("3.14 meter")
        assert abs(q.magnitude - 3.14) < 1e-10

    def test_parse_scientific_notation(self, ureg: UnitRegistry) -> None:
        q = ureg.parse_expression("3e-1 meter")
        assert abs(q.magnitude - 0.3) < 1e-10


class TestParseArithmeticExpressions:
    """Test parsing expressions with arithmetic operations."""

    def test_multiplication(self, ureg: UnitRegistry) -> None:
        q = ureg.parse_expression("6 meter")
        assert abs(q.magnitude - 6) < 1e-10

    def test_division(self, ureg: UnitRegistry) -> None:
        q = ureg.parse_expression("10 meter / second")
        assert abs(q.magnitude - 10) < 1e-10

    def test_exponentiation(self, ureg: UnitRegistry) -> None:
        q = ureg.parse_expression("3 meter ** 2")
        assert abs(q.magnitude - 3) < 1e-10


class TestParseUnitExpressions:
    """Test parsing unit expressions via parse_expression."""

    def test_simple_unit(self, ureg: UnitRegistry) -> None:
        q = ureg.parse_expression("9.8 m/s^2")
        assert abs(q.magnitude - 9.8) < 1e-10

    def test_compound_units(self, ureg: UnitRegistry) -> None:
        q = ureg.parse_expression("1 kg * m / s^2")
        assert abs(q.magnitude - 1.0) < 1e-10
        assert q.is_compatible_with("newton")

    def test_unit_with_prefix(self, ureg: UnitRegistry) -> None:
        q = ureg.parse_expression("5 km")
        result = q.to("meter")
        assert abs(result.magnitude - 5000) < 1e-10

    def test_negative_exponent(self, ureg: UnitRegistry) -> None:
        q = ureg.parse_expression("1 meter ** -1")
        assert abs(q.magnitude - 1) < 1e-10

    def test_multiple_units_with_exponents(self, ureg: UnitRegistry) -> None:
        q = ureg.parse_expression("1 kg ** 1 * s ** 2")
        assert abs(q.magnitude - 1) < 1e-10

    def test_gram_second_per_meter_squared(self, ureg: UnitRegistry) -> None:
        q = ureg.parse_expression("1 gram * second / meter ** 2")
        assert abs(q.magnitude - 1) < 1e-10


class TestQuantityStringParsing:
    """Test parsing through Quantity() constructor with strings."""

    def test_number_and_unit(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity("4.2 meter")
        assert abs(q.magnitude - 4.2) < 1e-10

    def test_number_star_unit(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity("4.2*meter")
        assert abs(q.magnitude - 4.2) < 1e-10

    def test_compound_string(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity("9.8 m/s^2")
        assert abs(q.magnitude - 9.8) < 1e-10

    def test_prefixed_unit_string(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity("5 km")
        result = q.to("meter")
        assert abs(result.magnitude - 5000) < 1e-10


class TestParseUnits:
    """Test ureg.parse_units()."""

    def test_simple_unit(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter")
        assert str(u) != ""

    def test_compound_unit(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("m/s")
        assert str(u) != ""

    def test_unit_with_exponent(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("m^2")
        s = str(u)
        assert "m" in s

    def test_unit_negative_exponent(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("m^-1")
        s = str(u)
        assert "m" in s

    def test_compound_multiple(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("kg * m / s^2")
        q = ureg.Quantity(1, str(u))
        assert q.is_compatible_with("newton")


class TestExpressionEvaluation:
    """Test that parsed expressions evaluate to correct values."""

    @pytest.mark.parametrize(
        ("expr", "expected_unit", "expected_mag"),
        [
            ("1 meter", "meter", 1.0),
            ("2.5 kilogram", "kilogram", 2.5),
            ("100 centimeter", "centimeter", 100.0),
            ("9.8 m/s^2", "m/s^2", 9.8),
        ],
    )
    def test_parse_and_check(
        self,
        ureg: UnitRegistry,
        expr: str,
        expected_unit: str,
        expected_mag: float,
    ) -> None:
        q = ureg.parse_expression(expr)
        assert abs(q.magnitude - expected_mag) < 1e-10

    @pytest.mark.parametrize(
        ("expr", "to_unit", "expected_mag"),
        [
            ("1 km", "meter", 1000.0),
            ("1 hour", "second", 3600.0),
            ("1 inch", "centimeter", 2.54),
        ],
    )
    def test_parse_and_convert(
        self,
        ureg: UnitRegistry,
        expr: str,
        to_unit: str,
        expected_mag: float,
    ) -> None:
        q = ureg.parse_expression(expr)
        result = q.to(to_unit)
        assert abs(result.magnitude - expected_mag) < 1e-6
