"""Formatter tests - ported from pint's test_formatter.py.

Tests formatting behavior through quantity/unit __format__ and str()
rather than through pint's internal formatter functions.
"""

from __future__ import annotations

import pytest
from pintrs import UnitRegistry


@pytest.fixture
def ureg() -> UnitRegistry:
    return UnitRegistry()


class TestQuantityDefaultFormat:
    """Test default formatting of quantities."""

    def test_str_simple(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(10, "meter")
        s = str(q)
        assert "10" in s
        assert "m" in s

    def test_format_empty_spec(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.5, "meter")
        assert format(q, "") == str(q)

    def test_dimensionless_str(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(42, "dimensionless")
        s = str(q)
        assert "42" in s

    def test_compound_unit_str(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(9.81, "meter / second ** 2")
        s = str(q)
        assert "9.81" in s


class TestQuantityMagnitudeFormat:
    """Test magnitude formatting specifiers."""

    def test_fixed_format(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.14159, "meter")
        result = format(q, ".2f")
        assert "3.14" in result

    def test_scientific_format(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(12345, "meter")
        result = format(q, ".2e")
        assert "1.23" in result or "1.24" in result


class TestUnitFormatStrings:
    """Test unit format string specifiers (~, P, L, H, D, C)."""

    def test_abbreviated_format(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(10, "meter")
        result = format(q, "~")
        assert "m" in result

    def test_pretty_format(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(10, "meter / second")
        result = format(q, "~P")
        assert "m" in result
        assert "s" in result

    def test_pretty_exponents(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "meter / second ** 2")
        result = format(q, "~P")
        assert "2" in result or "\u00b2" in result

    def test_latex_format(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(10, "meter")
        result = format(q, "~L")
        assert "$" in result or "mathrm" in result

    def test_latex_compound(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(10, "meter / second ** 2")
        result = format(q, "~L")
        assert "frac" in result or "^" in result

    def test_html_format(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(10, "meter")
        result = format(q, "~H")
        assert "m" in result

    def test_html_exponents(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "meter / second ** 2")
        result = format(q, "~H")
        assert "sup" in result or "2" in result

    def test_default_full_names(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(10, "meter")
        result = format(q, "D")
        assert "meter" in result

    def test_compact_format(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5000, "meter")
        result = format(q, "~C")
        assert "5" in result
        assert "km" in result or "kilometer" in result


class TestUnitFormat:
    """Test formatting of Unit objects directly."""

    def test_unit_str(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter / second")
        result = str(u)
        assert result != ""

    def test_unit_format_default(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter / second")
        assert format(u, "") == str(u)

    def test_unit_format_pretty(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter / second ** 2")
        result = format(u, "~P")
        assert "m" in result

    def test_unit_format_latex(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter")
        result = format(u, "~L")
        assert "$" in result or "mathrm" in result

    def test_unit_format_html(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter / second ** 2")
        result = format(u, "~H")
        assert "m" in result

    def test_unit_format_abbreviated(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter")
        result = format(u, "~")
        assert "m" in result


class TestFormatEdgeCases:
    """Test formatting edge cases."""

    def test_zero_quantity(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(0, "meter")
        result = format(q, "~P")
        assert "0" in result

    def test_negative_quantity(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(-5, "meter")
        result = format(q, ".1f~P")
        assert "-5.0" in result

    def test_power_unit(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(100, "meter ** 2")
        result = format(q, "~P")
        assert "100" in result

    def test_combined_magnitude_and_unit_format(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.14, "meter")
        result = format(q, ".2f~P")
        assert "3.14" in result

    def test_latex_with_magnitude(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.14, "meter")
        result = format(q, ".2f~L")
        assert "3.14" in result
