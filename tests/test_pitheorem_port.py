"""Pi theorem tests - ported from pint's test_pitheorem.py."""

from __future__ import annotations

import pytest
from pintrs import UnitRegistry


@pytest.fixture
def ureg() -> UnitRegistry:
    return UnitRegistry()


class TestPiTheorem:
    def test_simple_movement(self, ureg: UnitRegistry) -> None:
        result = ureg.pi_theorem({"V": "m/s", "T": "s", "L": "m"})
        assert len(result) == 1
        group = result[0]
        assert group["V"] == pytest.approx(1.0)
        assert group["T"] == pytest.approx(1.0)
        assert group["L"] == pytest.approx(-1.0)

    def test_pendulum(self, ureg: UnitRegistry) -> None:
        result = ureg.pi_theorem(
            {"T": "s", "M": "grams", "L": "m", "g": "m/s**2"}
        )
        assert len(result) == 1
        group = result[0]
        assert group["g"] == pytest.approx(1.0)
        assert group["T"] == pytest.approx(2.0)
        assert group["L"] == pytest.approx(-1.0)
        assert "M" not in group

    def test_empty_input(self, ureg: UnitRegistry) -> None:
        result = ureg.pi_theorem({})
        assert result == []

    def test_single_variable(self, ureg: UnitRegistry) -> None:
        result = ureg.pi_theorem({"L": "m"})
        assert result == []

    def test_two_compatible_variables(self, ureg: UnitRegistry) -> None:
        result = ureg.pi_theorem({"L1": "m", "L2": "m"})
        assert len(result) == 1
        group = result[0]
        assert abs(group.get("L1", 0)) + abs(group.get("L2", 0)) > 0

    def test_all_dimensionless(self, ureg: UnitRegistry) -> None:
        result = ureg.pi_theorem({"x": "dimensionless", "y": "dimensionless"})
        assert len(result) >= 1

    def test_fluid_dynamics(self, ureg: UnitRegistry) -> None:
        result = ureg.pi_theorem({
            "rho": "kg/m**3",
            "v": "m/s",
            "L": "m",
            "mu": "Pa*s",
        })
        assert len(result) == 1

    def test_heat_transfer(self, ureg: UnitRegistry) -> None:
        result = ureg.pi_theorem({
            "h": "W/(m**2*K)",
            "L": "m",
            "k": "W/(m*K)",
        })
        assert len(result) == 1

    def test_different_unit_styles(self, ureg: UnitRegistry) -> None:
        result1 = ureg.pi_theorem({"V": "m/s", "T": "s", "L": "m"})
        result2 = ureg.pi_theorem({"V": "km/hour", "T": "ms", "L": "cm"})
        assert len(result1) == len(result2)
