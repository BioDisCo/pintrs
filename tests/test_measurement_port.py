"""Measurement tests - ported from pint's test_measurement.py."""

from __future__ import annotations

import math

import pytest
from pintrs import Measurement, UnitRegistry


@pytest.fixture
def ureg() -> UnitRegistry:
    return UnitRegistry()


class TestMeasurementCreation:
    def test_from_quantity_and_error(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(10, "meter")
        m = Measurement(q, 0.5)
        assert m.value.magnitude == pytest.approx(10.0)
        assert m.error.magnitude == pytest.approx(0.5)

    def test_from_floats_and_units(self) -> None:
        m = Measurement(10.0, 0.5, "meter")
        assert m.value.magnitude == pytest.approx(10.0)
        assert m.error.magnitude == pytest.approx(0.5)
        assert "meter" in str(m.units) or "m" in str(m.units)

    def test_from_quantity_and_quantity_error(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(10, "meter")
        e = ureg.Quantity(0.5, "meter")
        m = Measurement(q, e)
        assert m.value.magnitude == pytest.approx(10.0)
        assert m.error.magnitude == pytest.approx(0.5)

    def test_zero_error(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(10, "meter")
        m = Measurement(q, 0)
        assert m.error.magnitude == pytest.approx(0.0)

    def test_plus_minus(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(10, "meter")
        m = q.plus_minus(0.5)
        assert m.value.magnitude == pytest.approx(10.0)
        assert m.error.magnitude == pytest.approx(0.5)


class TestMeasurementProperties:
    def test_magnitude(self, ureg: UnitRegistry) -> None:
        m = Measurement(ureg.Quantity(10, "meter"), 0.5)
        assert m.magnitude == pytest.approx(10.0)

    def test_units(self, ureg: UnitRegistry) -> None:
        m = Measurement(ureg.Quantity(10, "meter"), 0.5)
        assert "meter" in str(m.units) or "m" in str(m.units)

    def test_relative_error(self, ureg: UnitRegistry) -> None:
        m = Measurement(ureg.Quantity(100, "meter"), 10)
        assert m.rel == pytest.approx(0.1)

    def test_relative_error_zero_value(self, ureg: UnitRegistry) -> None:
        m = Measurement(ureg.Quantity(0, "meter"), 1)
        assert math.isinf(m.rel)


class TestMeasurementRepresentation:
    def test_str(self, ureg: UnitRegistry) -> None:
        m = Measurement(ureg.Quantity(10, "meter"), 0.5)
        s = str(m)
        assert "10" in s
        assert "0.5" in s

    def test_repr(self, ureg: UnitRegistry) -> None:
        m = Measurement(ureg.Quantity(10, "meter"), 0.5)
        r = repr(m)
        assert "10" in r
        assert "0.5" in r


class TestMeasurementArithmetic:
    def test_add(self, ureg: UnitRegistry) -> None:
        m1 = Measurement(ureg.Quantity(10, "meter"), 1)
        m2 = Measurement(ureg.Quantity(20, "meter"), 2)
        result = m1 + m2
        assert result.value.magnitude == pytest.approx(30.0)
        expected_err = math.sqrt(1**2 + 2**2)
        assert result.error.magnitude == pytest.approx(expected_err)

    def test_sub(self, ureg: UnitRegistry) -> None:
        m1 = Measurement(ureg.Quantity(30, "meter"), 2)
        m2 = Measurement(ureg.Quantity(10, "meter"), 1)
        result = m1 - m2
        assert result.value.magnitude == pytest.approx(20.0)
        expected_err = math.sqrt(2**2 + 1**2)
        assert result.error.magnitude == pytest.approx(expected_err)

    def test_mul_measurements(self, ureg: UnitRegistry) -> None:
        m1 = Measurement(ureg.Quantity(10, "meter"), 1)
        m2 = Measurement(ureg.Quantity(5, "second"), 0.5)
        result = m1 * m2
        assert result.value.magnitude == pytest.approx(50.0)

    def test_mul_scalar(self, ureg: UnitRegistry) -> None:
        m = Measurement(ureg.Quantity(10, "meter"), 1)
        result = m * 2
        assert result.value.magnitude == pytest.approx(20.0)
        assert result.error.magnitude == pytest.approx(2.0)

    def test_div_measurements(self, ureg: UnitRegistry) -> None:
        m1 = Measurement(ureg.Quantity(10, "meter"), 1)
        m2 = Measurement(ureg.Quantity(2, "second"), 0.1)
        result = m1 / m2
        assert result.value.magnitude == pytest.approx(5.0)

    def test_div_scalar(self, ureg: UnitRegistry) -> None:
        m = Measurement(ureg.Quantity(10, "meter"), 2)
        result = m / 2
        assert result.value.magnitude == pytest.approx(5.0)
        assert result.error.magnitude == pytest.approx(1.0)


class TestMeasurementComparison:
    def test_measurements_with_same_value(self, ureg: UnitRegistry) -> None:
        m1 = Measurement(ureg.Quantity(10, "meter"), 1)
        m2 = Measurement(ureg.Quantity(10, "meter"), 2)
        assert m1.value == m2.value

    def test_error_propagation_add_independent(self, ureg: UnitRegistry) -> None:
        m1 = Measurement(ureg.Quantity(100, "meter"), 3)
        m2 = Measurement(ureg.Quantity(200, "meter"), 4)
        result = m1 + m2
        assert result.error.magnitude == pytest.approx(5.0)

    def test_error_propagation_mul(self, ureg: UnitRegistry) -> None:
        m1 = Measurement(ureg.Quantity(10, "meter"), 1)
        m2 = Measurement(ureg.Quantity(20, "second"), 2)
        result = m1 * m2
        rel1 = 1 / 10
        rel2 = 2 / 20
        expected_rel = math.sqrt(rel1**2 + rel2**2)
        expected_abs = expected_rel * 200
        assert result.error.magnitude == pytest.approx(expected_abs)
