"""Logarithmic unit tests - ported from pint's test_log_units.py."""

from __future__ import annotations

import math

import pytest
from pintrs import UnitRegistry
from pintrs.logarithmic import LogarithmicQuantity


@pytest.fixture
def ureg() -> UnitRegistry:
    return UnitRegistry()


class TestLogQuantityCreation:
    def test_create_dB(self) -> None:
        lq = LogarithmicQuantity(10, "dB")
        assert lq.magnitude == pytest.approx(10.0)
        assert lq.units == "dB"

    def test_create_dBm(self) -> None:
        lq = LogarithmicQuantity(0, "dBm")
        assert lq.magnitude == pytest.approx(0.0)
        assert lq.units == "dBm"

    def test_create_dBW(self) -> None:
        lq = LogarithmicQuantity(30, "dBW")
        assert lq.magnitude == pytest.approx(30.0)

    def test_create_neper(self) -> None:
        lq = LogarithmicQuantity(1, "Np")
        assert lq.magnitude == pytest.approx(1.0)
        assert lq.units == "Np"

    def test_create_bel(self) -> None:
        lq = LogarithmicQuantity(1, "B")
        assert lq.magnitude == pytest.approx(1.0)


class TestLogConversions:
    def test_dB_to_linear(self) -> None:
        lq = LogarithmicQuantity(20, "dB")
        linear = lq.to_linear()
        assert linear == pytest.approx(100.0)

    def test_dB_zero(self) -> None:
        lq = LogarithmicQuantity(0, "dB")
        assert lq.to_linear() == pytest.approx(1.0)

    def test_dBm_to_mW(self) -> None:
        lq = LogarithmicQuantity(0, "dBm")
        assert lq.to_linear() == pytest.approx(1.0)

    def test_dBm_30_to_mW(self) -> None:
        lq = LogarithmicQuantity(30, "dBm")
        assert lq.to_linear() == pytest.approx(1000.0)

    def test_dBm_minus10_to_mW(self) -> None:
        lq = LogarithmicQuantity(-10, "dBm")
        assert lq.to_linear() == pytest.approx(0.1)

    def test_neper_to_linear(self) -> None:
        lq = LogarithmicQuantity(1, "Np")
        assert lq.to_linear() == pytest.approx(math.e**2)

    def test_bel_to_linear(self) -> None:
        lq = LogarithmicQuantity(1, "B")
        assert lq.to_linear() == pytest.approx(10.0)

    def test_dBW_to_linear(self) -> None:
        lq = LogarithmicQuantity(0, "dBW")
        assert lq.to_linear() == pytest.approx(1.0)

    def test_dBW_30(self) -> None:
        lq = LogarithmicQuantity(30, "dBW")
        assert lq.to_linear() == pytest.approx(1000.0)

    @pytest.mark.parametrize(
        ("db_val", "expected_linear"),
        [
            (0, 1.0),
            (10, 10.0),
            (20, 100.0),
            (30, 1000.0),
            (-10, 0.1),
            (-20, 0.01),
            (3, pytest.approx(2.0, rel=0.01)),
            (6, pytest.approx(4.0, rel=0.01)),
        ],
    )
    def test_dB_parametrized(self, db_val: float, expected_linear: float) -> None:
        lq = LogarithmicQuantity(db_val, "dB")
        assert lq.to_linear() == pytest.approx(expected_linear, rel=0.02)


class TestLogFromLinear:
    def test_from_linear_dB(self) -> None:
        lq = LogarithmicQuantity.from_linear(100, "dB")
        assert lq.magnitude == pytest.approx(20.0)

    def test_from_linear_dBm(self) -> None:
        lq = LogarithmicQuantity.from_linear(1, "dBm")
        assert lq.magnitude == pytest.approx(0.0)

    def test_from_linear_neper(self) -> None:
        lq = LogarithmicQuantity.from_linear(math.e**2, "Np")
        assert lq.magnitude == pytest.approx(1.0)

    def test_from_linear_bel(self) -> None:
        lq = LogarithmicQuantity.from_linear(10, "B")
        assert lq.magnitude == pytest.approx(1.0)


class TestLogArithmetic:
    def test_add_dB(self) -> None:
        a = LogarithmicQuantity(10, "dB")
        b = LogarithmicQuantity(10, "dB")
        result = a + b
        assert result.magnitude == pytest.approx(20.0)

    def test_sub_dB(self) -> None:
        a = LogarithmicQuantity(20, "dB")
        b = LogarithmicQuantity(10, "dB")
        result = a - b
        assert result.magnitude == pytest.approx(10.0)

    def test_mul_scalar(self) -> None:
        a = LogarithmicQuantity(10, "dB")
        result = a * 2
        assert result.magnitude == pytest.approx(20.0)

    def test_div_scalar(self) -> None:
        a = LogarithmicQuantity(20, "dB")
        result = a / 2
        assert result.magnitude == pytest.approx(10.0)

    def test_neg(self) -> None:
        a = LogarithmicQuantity(10, "dB")
        result = -a
        assert result.magnitude == pytest.approx(-10.0)


class TestLogComparison:
    def test_eq(self) -> None:
        a = LogarithmicQuantity(10, "dB")
        b = LogarithmicQuantity(10, "dB")
        assert a == b

    def test_ne(self) -> None:
        a = LogarithmicQuantity(10, "dB")
        b = LogarithmicQuantity(20, "dB")
        assert a != b

    def test_lt(self) -> None:
        a = LogarithmicQuantity(10, "dB")
        b = LogarithmicQuantity(20, "dB")
        assert a < b

    def test_gt(self) -> None:
        a = LogarithmicQuantity(20, "dB")
        b = LogarithmicQuantity(10, "dB")
        assert a > b

    def test_le(self) -> None:
        a = LogarithmicQuantity(10, "dB")
        b = LogarithmicQuantity(10, "dB")
        assert a <= b

    def test_ge(self) -> None:
        a = LogarithmicQuantity(10, "dB")
        b = LogarithmicQuantity(10, "dB")
        assert a >= b


class TestLogRepresentation:
    def test_str(self) -> None:
        lq = LogarithmicQuantity(10, "dB")
        s = str(lq)
        assert "10" in s
        assert "dB" in s

    def test_repr(self) -> None:
        lq = LogarithmicQuantity(10, "dB")
        r = repr(lq)
        assert "10" in r
        assert "dB" in r


class TestLogUnitConversion:
    def test_dB_to_Np(self) -> None:
        lq = LogarithmicQuantity(20, "dB")
        converted = lq.to("Np")
        linear_orig = lq.to_linear()
        linear_conv = converted.to_linear()
        assert linear_orig == pytest.approx(linear_conv, rel=1e-6)

    def test_Np_to_dB(self) -> None:
        lq = LogarithmicQuantity(1, "Np")
        converted = lq.to("dB")
        linear_orig = lq.to_linear()
        linear_conv = converted.to_linear()
        assert linear_orig == pytest.approx(linear_conv, rel=1e-6)

    def test_dB_to_B(self) -> None:
        lq = LogarithmicQuantity(20, "dB")
        converted = lq.to("B")
        assert converted.magnitude == pytest.approx(2.0)

    def test_B_to_dB(self) -> None:
        lq = LogarithmicQuantity(2, "B")
        converted = lq.to("dB")
        assert converted.magnitude == pytest.approx(20.0)

    def test_dBm_to_dBW(self) -> None:
        lq = LogarithmicQuantity(30, "dBm")
        converted = lq.to("dBW")
        assert converted.magnitude == pytest.approx(0.0)

    def test_dBW_to_dBm(self) -> None:
        lq = LogarithmicQuantity(0, "dBW")
        converted = lq.to("dBm")
        assert converted.magnitude == pytest.approx(30.0)
