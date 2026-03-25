"""Tests for unit formatting modes."""

from __future__ import annotations

import pytest
from pintrs import UnitRegistry


@pytest.fixture
def ureg() -> UnitRegistry:
    return UnitRegistry()


class TestPrettyFormat:
    def test_simple(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(9.81, "meter / second ** 2")
        result = format(q, "~P")
        assert "²" in result
        assert "m" in result

    def test_full_names(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(9.81, "meter / second ** 2")
        result = format(q, "P")
        assert "second²" in result

    def test_mul_separator(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "kilogram * meter / second ** 2")
        result = format(q, "P")
        assert "·" in result


class TestLatexFormat:
    def test_simple(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(9.81, "meter / second ** 2")
        result = format(q, "~L")
        assert r"\frac" in result
        assert "$" in result

    def test_with_magnitude_format(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(9.81, "meter")
        result = format(q, ".2f~L")
        assert "9.81" in result
        assert "$" in result


class TestHtmlFormat:
    def test_simple(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(9.81, "meter / second ** 2")
        result = format(q, "~H")
        assert "<sup>" in result

    def test_full_names(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(9.81, "meter / second ** 2")
        result = format(q, "H")
        assert "second<sup>2</sup>" in result


class TestCompactFormat:
    def test_large_value(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1500.0, "meter")
        result = format(q, "~C")
        assert "1.5" in result

    def test_with_magnitude_format(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1500.0, "meter")
        result = format(q, ".1f~C")
        assert "1.5" in result


class TestDefaultFormat:
    def test_empty_spec(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(9.81, "meter / second ** 2")
        assert format(q, "") == str(q)

    def test_magnitude_format(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(9.81, "meter")
        result = format(q, ".1f")
        assert "9.8" in result

    def test_dimensionless(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.14, "dimensionless")
        result = format(q, "~P")
        assert "3.14" in result


class TestUnitFormat:
    def test_unit_pretty(self, ureg: UnitRegistry) -> None:
        from pintrs.formatting import format_unit

        u = ureg.Unit("meter / second ** 2")
        result = format_unit(u, "~P")
        assert "²" in result

    def test_unit_latex(self, ureg: UnitRegistry) -> None:
        from pintrs.formatting import format_unit

        u = ureg.Unit("meter / second ** 2")
        result = format_unit(u, "~L")
        assert r"\frac" in result

    def test_unit_html(self, ureg: UnitRegistry) -> None:
        from pintrs.formatting import format_unit

        u = ureg.Unit("meter / second ** 2")
        result = format_unit(u, "~H")
        assert "<sup>" in result
