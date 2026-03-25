"""Babel formatting tests - ported from pint's test_babel.py."""

from __future__ import annotations

import pytest

babel = pytest.importorskip("babel")

from pintrs import UnitRegistry


@pytest.fixture
def ureg() -> UnitRegistry:
    return UnitRegistry()


class TestBabelFormatting:
    def test_format_default_locale(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1000.5, "meter")
        result = q.format_babel("en")
        assert "1" in result
        assert "000" in result or "," in result

    def test_format_german_locale(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1000.5, "meter")
        result = q.format_babel("de_DE")
        assert "1" in result

    def test_format_french_locale(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1000.5, "meter")
        result = q.format_babel("fr_FR")
        assert "1" in result

    def test_unit_format_babel(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter")
        result = u.format_babel("en")
        assert "meter" in result or "m" in result

    def test_integer_quantity(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(42, "second")
        result = q.format_babel("en")
        assert "42" in result

    def test_zero_quantity(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(0, "meter")
        result = q.format_babel("en")
        assert "0" in result

    def test_negative_quantity(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(-5, "meter")
        result = q.format_babel("en")
        assert "-5" in result

    def test_large_number(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1_000_000, "meter")
        result = q.format_babel("en_US")
        assert "1" in result
        assert "000" in result


class TestBabelNoLocale:
    def test_no_babel_fallback(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(10, "meter")
        result = q.format_babel()
        assert "10" in result
