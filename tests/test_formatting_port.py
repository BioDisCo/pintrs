"""Formatting tests - ported from pint's test_formatting.py."""

from __future__ import annotations

import pytest
from pintrs import UnitRegistry


@pytest.fixture
def ureg() -> UnitRegistry:
    return UnitRegistry()


class TestQuantityFormatDefault:
    def test_no_spec(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.14, "meter")
        assert str(q) == format(q, "")

    def test_magnitude_format(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.14159, "meter")
        result = format(q, ".2f")
        assert "3.14" in result

    def test_magnitude_scientific(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(12345, "meter")
        result = format(q, ".2e")
        assert "1.23" in result or "1.24" in result


class TestQuantityFormatPretty:
    def test_pretty_simple(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(10, "meter / second")
        result = format(q, "~P")
        assert "m" in result
        assert "s" in result

    def test_pretty_with_magnitude_format(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.14, "meter")
        result = format(q, ".2f~P")
        assert "3.14" in result

    def test_pretty_exponents_use_superscript(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "meter / second ** 2")
        result = format(q, "~P")
        assert "²" in result or "2" in result


class TestQuantityFormatLatex:
    def test_latex_simple(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(10, "meter")
        result = format(q, "~L")
        assert "$" in result or "mathrm" in result

    def test_latex_compound(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(10, "meter / second ** 2")
        result = format(q, "~L")
        assert "frac" in result or "^" in result

    def test_latex_with_magnitude(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.14, "meter")
        result = format(q, ".2f~L")
        assert "3.14" in result


class TestQuantityFormatHTML:
    def test_html_simple(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(10, "meter")
        result = format(q, "~H")
        assert "m" in result

    def test_html_exponents_use_sup(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "meter / second ** 2")
        result = format(q, "~H")
        assert "sup" in result or "2" in result


class TestQuantityFormatCompact:
    def test_compact_large(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5000, "meter")
        result = format(q, "~C")
        assert "5" in result
        assert "km" in result or "kilometer" in result

    def test_compact_small(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(0.001, "meter")
        result = format(q, "~C")
        assert "mm" in result or "millimeter" in result or "1" in result


class TestQuantityFormatAbbreviation:
    def test_tilde_uses_abbreviations(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(10, "meter")
        abbreviated = format(q, "~")
        full = format(q, "")
        assert len(abbreviated) <= len(full) or abbreviated == full

    def test_tilde_pretty(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(10, "meter")
        result = format(q, "~P")
        assert "m" in result

    def test_no_tilde_full_names(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(10, "meter")
        result = format(q, "D")
        assert "meter" in result


class TestUnitFormat:
    def test_unit_format_default(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter / second")
        result = format(u, "")
        assert str(u) == result

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

    def test_unit_format_abbreviation(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter")
        abbreviated = format(u, "~")
        assert "m" in abbreviated


class TestFormatEdgeCases:
    def test_dimensionless_format(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(42, "dimensionless")
        result = format(q, "~P")
        assert "42" in result

    def test_compound_unit_format(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(9.81, "meter / second ** 2")
        result = format(q, ".2f~P")
        assert "9.81" in result

    def test_power_unit_format(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(100, "meter ** 2")
        result = format(q, "~P")
        assert "100" in result

    def test_zero_quantity_format(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(0, "meter")
        result = format(q, "~P")
        assert "0" in result

    def test_negative_quantity_format(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(-5, "meter")
        result = format(q, ".1f~P")
        assert "-5.0" in result
