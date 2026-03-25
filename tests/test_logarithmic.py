"""Tests for logarithmic units."""

from __future__ import annotations

import pytest
from pintrs import LogarithmicQuantity, UnitRegistry


@pytest.fixture
def ureg() -> UnitRegistry:
    return UnitRegistry()


class TestLogarithmicCreation:
    def test_create_dB(self) -> None:
        q = LogarithmicQuantity(3.0, "dB")
        assert q.magnitude == 3.0
        assert q.units == "dB"

    def test_create_dBm(self) -> None:
        q = LogarithmicQuantity(30.0, "dBm")
        assert q.magnitude == 30.0
        assert q.units == "dBm"

    def test_create_Np(self) -> None:
        q = LogarithmicQuantity(1.0, "Np")
        assert q.units == "Np"

    def test_invalid_unit(self) -> None:
        with pytest.raises(ValueError, match="Unknown logarithmic unit"):
            LogarithmicQuantity(1.0, "dBx")


class TestLinearConversion:
    def test_dB_to_linear(self) -> None:
        q = LogarithmicQuantity(10.0, "dB")
        assert abs(q.to_linear() - 10.0) < 1e-10

    def test_dB_to_linear_20(self) -> None:
        q = LogarithmicQuantity(20.0, "dB")
        assert abs(q.to_linear() - 100.0) < 1e-10

    def test_dB_zero(self) -> None:
        q = LogarithmicQuantity(0.0, "dB")
        assert abs(q.to_linear() - 1.0) < 1e-10

    def test_dBm_to_linear(self) -> None:
        q = LogarithmicQuantity(30.0, "dBm")
        # 30 dBm = 1 W = 1000 mW
        assert abs(q.to_linear() - 1000.0) < 1e-6

    def test_from_linear(self) -> None:
        q = LogarithmicQuantity.from_linear(100.0, "dB")
        assert abs(q.magnitude - 20.0) < 1e-10

    def test_from_linear_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="non-positive"):
            LogarithmicQuantity.from_linear(-1.0, "dB")


class TestLogarithmicConversion:
    def test_dB_to_Bel(self) -> None:
        q = LogarithmicQuantity(20.0, "dB")
        result = q.to("Bel")
        assert abs(result.magnitude - 2.0) < 1e-10

    def test_dBm_to_dBW(self) -> None:
        q = LogarithmicQuantity(30.0, "dBm")
        result = q.to("dBW")
        assert abs(result.magnitude - 0.0) < 1e-10


class TestToQuantity:
    def test_dBm_to_quantity(self, ureg: UnitRegistry) -> None:
        q = LogarithmicQuantity(30.0, "dBm", ureg)
        result = q.to_quantity()
        # 30 dBm = 1 W
        assert abs(result.magnitude - 1.0) < 1e-6
        assert str(result.units) == "W"

    def test_dBW_to_quantity(self, ureg: UnitRegistry) -> None:
        q = LogarithmicQuantity(0.0, "dBW", ureg)
        result = q.to_quantity()
        assert abs(result.magnitude - 1.0) < 1e-6

    def test_plain_dB_raises(self) -> None:
        q = LogarithmicQuantity(10.0, "dB")
        with pytest.raises(ValueError, match="relative unit"):
            q.to_quantity()


class TestLogarithmicArithmetic:
    def test_add(self) -> None:
        a = LogarithmicQuantity(10.0, "dB")
        b = LogarithmicQuantity(3.0, "dB")
        result = a + b
        assert abs(result.magnitude - 13.0) < 1e-10

    def test_sub(self) -> None:
        a = LogarithmicQuantity(20.0, "dBm")
        b = LogarithmicQuantity(10.0, "dBm")
        result = a - b
        assert abs(result.magnitude - 10.0) < 1e-10

    def test_add_scalar(self) -> None:
        q = LogarithmicQuantity(10.0, "dB")
        result = q + 3.0
        assert abs(result.magnitude - 13.0) < 1e-10

    def test_neg(self) -> None:
        q = LogarithmicQuantity(10.0, "dB")
        assert (-q).magnitude == -10.0

    def test_eq(self) -> None:
        a = LogarithmicQuantity(10.0, "dB")
        b = LogarithmicQuantity(10.0, "dB")
        assert a == b

    def test_lt(self) -> None:
        a = LogarithmicQuantity(3.0, "dB")
        b = LogarithmicQuantity(10.0, "dB")
        assert a < b

    def test_repr(self) -> None:
        q = LogarithmicQuantity(10.0, "dBm")
        assert "10.0" in repr(q)
        assert "dBm" in repr(q)

    def test_str(self) -> None:
        q = LogarithmicQuantity(10.0, "dBm")
        assert str(q) == "10.0 dBm"
