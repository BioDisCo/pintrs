"""Converter tests - ported from pint's test_converters.py.

Tests conversion behavior (scale, offset, logarithmic) through the
quantity API rather than through pint's internal converter classes.
"""

from __future__ import annotations

import math

import pytest
from pintrs import UnitRegistry


@pytest.fixture
def ureg() -> UnitRegistry:
    return UnitRegistry()


class TestScaleConversion:
    """Test multiplicative/scale conversions (mirrors ScaleConverter tests)."""

    def test_simple_scale(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "kilometer")
        result = q.to("meter")
        assert abs(result.magnitude - 1000) < 1e-10

    def test_roundtrip_scale(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(100, "centimeter")
        roundtrip = q.to("meter").to("centimeter")
        assert abs(roundtrip.magnitude - 100) < 1e-10

    def test_prefix_scale(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "milligram")
        result = q.to("gram")
        assert abs(result.magnitude - 0.001) < 1e-10

    def test_micro_prefix(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1e6, "micrometer")
        result = q.to("meter")
        assert abs(result.magnitude - 1.0) < 1e-10

    def test_compound_scale(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "kilometer / hour")
        result = q.to("meter / second")
        assert abs(result.magnitude - (1000 / 3600)) < 1e-10


class TestOffsetConversion:
    """Test offset conversions like temperature (mirrors OffsetConverter tests)."""

    def test_celsius_to_kelvin(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(0, "degree_Celsius")
        result = q.to("kelvin")
        assert abs(result.magnitude - 273.15) < 1e-6

    def test_kelvin_to_celsius(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(273.15, "kelvin")
        result = q.to("degree_Celsius")
        assert abs(result.magnitude) < 1e-6

    def test_celsius_to_fahrenheit(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(100, "degree_Celsius")
        result = q.to("degree_Fahrenheit")
        assert abs(result.magnitude - 212.0) < 1e-6

    def test_fahrenheit_to_celsius(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(32, "degree_Fahrenheit")
        result = q.to("degree_Celsius")
        assert abs(result.magnitude) < 1e-6

    def test_roundtrip_celsius_kelvin(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(100, "degree_Celsius")
        roundtrip = q.to("kelvin").to("degree_Celsius")
        assert abs(roundtrip.magnitude - 100) < 1e-6

    def test_roundtrip_fahrenheit_celsius(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(72, "degree_Fahrenheit")
        roundtrip = q.to("degree_Celsius").to("degree_Fahrenheit")
        assert abs(roundtrip.magnitude - 72) < 1e-6

    def test_absolute_zero_kelvin(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(0, "kelvin")
        result = q.to("degree_Celsius")
        assert abs(result.magnitude - (-273.15)) < 1e-6

    def test_absolute_zero_fahrenheit(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(0, "kelvin")
        result = q.to("degree_Fahrenheit")
        assert abs(result.magnitude - (-459.67)) < 1e-4


class TestLogarithmicConversion:
    """Test logarithmic conversions (mirrors LogarithmicConverter tests)."""

    def test_dBm_0_is_1mW(self, ureg: UnitRegistry) -> None:
        from pintrs import LogarithmicQuantity

        lq = LogarithmicQuantity(0, "dBm")
        linear = lq.to_linear()
        assert linear == pytest.approx(1.0)

    def test_dBm_30_is_1W(self, ureg: UnitRegistry) -> None:
        from pintrs import LogarithmicQuantity

        lq = LogarithmicQuantity(30, "dBm")
        linear = lq.to_linear()
        assert linear == pytest.approx(1000.0)

    def test_dB_0_is_unity(self, ureg: UnitRegistry) -> None:
        from pintrs import LogarithmicQuantity

        lq = LogarithmicQuantity(0, "dB")
        assert lq.to_linear() == pytest.approx(1.0)

    def test_dB_10_is_10(self, ureg: UnitRegistry) -> None:
        from pintrs import LogarithmicQuantity

        lq = LogarithmicQuantity(10, "dB")
        assert lq.to_linear() == pytest.approx(10.0)

    def test_dB_20_is_100(self, ureg: UnitRegistry) -> None:
        from pintrs import LogarithmicQuantity

        lq = LogarithmicQuantity(20, "dB")
        assert lq.to_linear() == pytest.approx(100.0)

    def test_dB_roundtrip(self, ureg: UnitRegistry) -> None:
        from pintrs import LogarithmicQuantity

        arb_value = 20.0
        lq = LogarithmicQuantity(arb_value, "dB")
        linear = lq.to_linear()
        back = LogarithmicQuantity.from_linear(linear, "dB")
        assert back.magnitude == pytest.approx(arb_value)

    def test_neper_to_linear(self, ureg: UnitRegistry) -> None:
        from pintrs import LogarithmicQuantity

        lq = LogarithmicQuantity(1, "Np")
        assert lq.to_linear() == pytest.approx(math.e**2)

    def test_bel_to_linear(self, ureg: UnitRegistry) -> None:
        from pintrs import LogarithmicQuantity

        lq = LogarithmicQuantity(1, "B")
        assert lq.to_linear() == pytest.approx(10.0)

    def test_from_linear_dB(self, ureg: UnitRegistry) -> None:
        from pintrs import LogarithmicQuantity

        lq = LogarithmicQuantity.from_linear(100, "dB")
        assert lq.magnitude == pytest.approx(20.0)

    def test_from_linear_dBm(self, ureg: UnitRegistry) -> None:
        from pintrs import LogarithmicQuantity

        lq = LogarithmicQuantity.from_linear(1, "dBm")
        assert lq.magnitude == pytest.approx(0.0)


class TestScaleConversionNumpy:
    """Test scale conversions with numpy arrays."""

    def test_array_scale_roundtrip(self, ureg: UnitRegistry) -> None:
        np = pytest.importorskip("numpy")
        from pintrs import ArrayQuantity

        a = ArrayQuantity(np.ones((1, 10)), "kilometer", ureg)
        result = a.to("meter").to("kilometer")
        np.testing.assert_allclose(result.magnitude, np.ones((1, 10)))
