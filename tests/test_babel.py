"""Tests for Babel/locale formatting."""

from __future__ import annotations

import pytest
from pintrs import UnitRegistry

babel = pytest.importorskip("babel")


@pytest.fixture
def ureg() -> UnitRegistry:
    return UnitRegistry()


class TestBabelFormatting:
    def test_format_babel_default(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1000.5, "meter")
        result = q.format_babel()
        assert "1" in result
        assert "m" in result

    def test_format_babel_german(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1000.5, "meter")
        result = q.format_babel("de_DE")
        # German uses comma for decimal
        assert "," in result

    def test_format_babel_french(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1000.5, "meter")
        result = q.format_babel("fr_FR")
        assert "m" in result

    def test_unit_format_babel(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("meter")
        result = u.format_babel("en_US")
        assert isinstance(result, str)

    def test_format_babel_without_babel(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "meter")
        # Even if babel is installed, calling with default should work
        result = q.format_babel("en")
        assert "5" in result
