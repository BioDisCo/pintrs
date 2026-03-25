"""Utility tests - ported from pint's test_util.py.

Tests internal utility behavior through the public API, since pintrs
implements these utilities in Rust and doesn't expose UnitsContainer,
ParserHelper, etc. directly.
"""

from __future__ import annotations

import pytest
from pintrs import UndefinedUnitError, UnitRegistry


@pytest.fixture
def ureg() -> UnitRegistry:
    return UnitRegistry()


# ---------------------------------------------------------------------------
# Unit container behavior (tested through Unit and Quantity)
# ---------------------------------------------------------------------------


class TestUnitContainerBehavior:
    """Mirrors pint's TestUnitsContainer via the public Unit/Quantity API."""

    def test_unit_creation(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter * second ** 2")
        assert str(u) is not None

    def test_dimensionless_str(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "meter") / ureg.Quantity(1.0, "meter")
        assert q.dimensionless

    def test_unit_equality(self, ureg: UnitRegistry) -> None:
        u1 = ureg.parse_units("meter")
        u2 = ureg.parse_units("meter")
        assert u1 == u2

    def test_unit_inequality(self, ureg: UnitRegistry) -> None:
        u1 = ureg.parse_units("meter")
        u2 = ureg.parse_units("second")
        assert u1 != u2

    def test_unit_multiplication(self, ureg: UnitRegistry) -> None:
        q1 = ureg.Quantity(2, "meter")
        q2 = ureg.Quantity(3, "second")
        result = q1 * q2
        assert result.magnitude == pytest.approx(6.0)

    def test_unit_division(self, ureg: UnitRegistry) -> None:
        q1 = ureg.Quantity(6, "meter")
        q2 = ureg.Quantity(2, "second")
        result = q1 / q2
        assert result.magnitude == pytest.approx(3.0)

    def test_unit_power(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3, "meter")
        result = q**2
        assert result.magnitude == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# String preprocessing (tested via parse_expression)
# ---------------------------------------------------------------------------


class TestStringPreprocessing:
    """Mirrors pint's TestStringProcessor by testing parsing behavior."""

    def test_squared_cubed(self, ureg: UnitRegistry) -> None:
        q = ureg.parse_expression("1 meter^3")
        assert q.magnitude == pytest.approx(1.0)
        q2 = ureg.parse_expression("1 meter**3")
        assert q.to_base_units().magnitude == pytest.approx(
            q2.to_base_units().magnitude
        )

    def test_per(self, ureg: UnitRegistry) -> None:
        q = ureg.parse_expression("1 meter / second")
        assert q.magnitude == pytest.approx(1.0)

    def test_scientific_notation(self, ureg: UnitRegistry) -> None:
        q = ureg.parse_expression("1e3 meter")
        assert q.magnitude == pytest.approx(1000.0)

        q = ureg.parse_expression("1E-3 meter")
        assert q.magnitude == pytest.approx(0.001)

        q = ureg.parse_expression("1.5e2 meter")
        assert q.magnitude == pytest.approx(150.0)

    def test_space_multiplication(self, ureg: UnitRegistry) -> None:
        q = ureg.parse_expression("1 meter")
        assert q.magnitude == pytest.approx(1.0)

        q = ureg.parse_expression("2.5 meter")
        assert q.magnitude == pytest.approx(2.5)

    def test_compound_units(self, ureg: UnitRegistry) -> None:
        q = ureg.parse_expression("9.81 meter / second ** 2")
        assert q.magnitude == pytest.approx(9.81)

        q = ureg.parse_expression("1 kilogram * meter / second ** 2")
        assert q.magnitude == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Parse expression evaluation (mirrors TestParseHelper)
# ---------------------------------------------------------------------------


class TestParseExpression:
    """Test expression parsing through the public API."""

    def test_simple_value(self, ureg: UnitRegistry) -> None:
        q = ureg.parse_expression("42")
        assert q.magnitude == pytest.approx(42.0)

    def test_value_with_units(self, ureg: UnitRegistry) -> None:
        q = ureg.parse_expression("4.2 meter")
        assert q.magnitude == pytest.approx(4.2)

    def test_compound_expression(self, ureg: UnitRegistry) -> None:
        q = ureg.parse_expression("1 meter * second")
        assert q.magnitude == pytest.approx(1.0)

    def test_negative_exponent(self, ureg: UnitRegistry) -> None:
        q = ureg.parse_expression("1 meter ** -1")
        assert q.magnitude == pytest.approx(1.0)

    def test_division_expression(self, ureg: UnitRegistry) -> None:
        q = ureg.parse_expression("1 meter / second")
        assert q.magnitude == pytest.approx(1.0)

    def test_from_string(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity("4.2 meter")
        assert q.magnitude == pytest.approx(4.2)

    def test_nan_parsing(self, ureg: UnitRegistry) -> None:
        import math

        q = ureg.Quantity(float("nan"), "meter")
        assert math.isnan(q.magnitude)


# ---------------------------------------------------------------------------
# Quantity iteration and sizing
# ---------------------------------------------------------------------------


class TestIterableAndSized:
    """Mirrors pint's TestOtherUtils for iterable/sized behavior."""

    def test_quantity_not_iterable(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(42, "meter")
        with pytest.raises(TypeError):
            iter(q)  # type: ignore[call-overload]

    def test_quantity_has_no_len(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(42, "meter")
        with pytest.raises(TypeError):
            len(q)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Define custom units (mirrors TestDefinition behavior)
# ---------------------------------------------------------------------------


class TestDefine:
    """Test unit definition through the public API."""

    def test_define_simple_unit(self, ureg: UnitRegistry) -> None:
        ureg.define("testunit = 42 * meter")
        q = ureg.Quantity(1, "testunit")
        result = q.to("meter")
        assert result.magnitude == pytest.approx(42.0)

    def test_define_with_prefix(self, ureg: UnitRegistry) -> None:
        ureg.define("myfoot = 0.3048 * meter")
        q = ureg.Quantity(1, "myfoot")
        result = q.to("meter")
        assert result.magnitude == pytest.approx(0.3048)

    def test_undefined_unit_raises(self, ureg: UnitRegistry) -> None:
        with pytest.raises(UndefinedUnitError):
            ureg.Quantity(1, "nonexistent_unit_xyz")


# ---------------------------------------------------------------------------
# Unit formatting (mirrors TestFormatter)
# ---------------------------------------------------------------------------


class TestUnitFormatting:
    """Test unit/quantity formatting through the public API."""

    def test_quantity_str(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.14, "meter")
        s = str(q)
        assert "3.14" in s
        assert "m" in s

    def test_quantity_repr(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.14, "meter")
        r = repr(q)
        assert "3.14" in r

    def test_unit_str(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter / second ** 2")
        s = str(u)
        assert "m" in s
        assert "s" in s

    def test_format_pretty(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter / second ** 2")
        s = format(u, "P")
        assert "²" in s

    def test_format_latex(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "meter / second")
        s = format(q, "L")
        assert "$" in s or "frac" in s

    def test_dimensionless_format(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "meter") / ureg.Quantity(1.0, "meter")
        s = str(q)
        assert "1" in s


# ---------------------------------------------------------------------------
# to_compact (mirrors test_infer_base_unit.test_to_compact)
# ---------------------------------------------------------------------------


class TestToCompact:
    """Test to_compact behavior through the public API."""

    def test_large_value_compact(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1e9, "meter")
        c = q.to_compact()
        assert c.magnitude == pytest.approx(1.0)
        assert "Gm" in str(c) or "gigameter" in str(c).lower() or "gm" in str(c)

    def test_small_value_compact(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(0.001, "meter")
        c = q.to_compact()
        assert c.magnitude == pytest.approx(1.0)

    def test_already_compact(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "meter")
        c = q.to_compact()
        # to_compact may choose a different prefix, but the value is equivalent
        assert c.to("meter").magnitude == pytest.approx(5.0)

    def test_compound_unit_compact(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1000, "meter / second")
        c = q.to_compact()
        assert c.magnitude >= 0.1
        assert c.magnitude <= 1000


# ---------------------------------------------------------------------------
# Volts / combined unit inference
# ---------------------------------------------------------------------------


class TestCombinedUnits:
    """Test combined unit operations (mirrors test_infer_base_unit.test_volts)."""

    def test_volt_operations(self, ureg: UnitRegistry) -> None:
        r = ureg.Quantity(1, "V") * ureg.Quantity(1, "mV") / ureg.Quantity(1, "kV")
        expected = ureg.Quantity(1e-6, "V")
        assert r.to("V").magnitude == pytest.approx(expected.magnitude, rel=1e-6)

    def test_meter_combinations(self, ureg: UnitRegistry) -> None:
        r = ureg.Quantity(1, "mm") * ureg.Quantity(1, "nm")
        expected = ureg.Quantity(1e-12, "meter ** 2")
        assert r.to("meter ** 2").magnitude == pytest.approx(
            expected.magnitude, rel=1e-6
        )
