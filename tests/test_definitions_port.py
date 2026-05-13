"""Definition tests - ported from pint's test_definitions.py.

Tests unit definition behavior through ureg.define() and verifying
that defined units work correctly, rather than testing pint's internal
Definition/PrefixDefinition/UnitDefinition classes.
"""

from __future__ import annotations

import pytest
from pintrs import DimensionalityError, UnitRegistry


@pytest.fixture
def ureg() -> UnitRegistry:
    return UnitRegistry()


class TestDefineCustomUnits:
    """Test defining custom units via ureg.define()."""

    def test_define_simple_unit(self, ureg: UnitRegistry) -> None:
        ureg.define("smoot = 1.7018 * meter")
        q = ureg.Quantity(1, "smoot").to("meter")
        assert abs(q.magnitude - 1.7018) < 1e-10

    def test_define_and_convert_back(self, ureg: UnitRegistry) -> None:
        ureg.define("smoot = 1.7018 * meter")
        q = ureg.Quantity(1.7018, "meter").to("smoot")
        assert abs(q.magnitude - 1.0) < 1e-6

    def test_define_compound_unit(self, ureg: UnitRegistry) -> None:
        ureg.define("speed_unit = kilometer / hour")
        q = ureg.Quantity(1, "speed_unit").to("meter / second")
        expected = 1000.0 / 3600.0
        assert abs(q.magnitude - expected) < 1e-10

    def test_define_with_scale(self, ureg: UnitRegistry) -> None:
        ureg.define("myunit = 96485.3399 * coulomb")
        q = ureg.Quantity(1, "myunit").to("coulomb")
        assert abs(q.magnitude - 96485.3399) < 1e-4


class TestBuiltinUnitDefinitions:
    """Verify built-in unit definitions work correctly."""

    def test_meter_is_base_length(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "meter")
        assert q.dimensionality == "[length]"

    def test_coulomb_definition(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "coulomb")
        base = q.to_base_units()
        assert "second" in str(base.units)
        assert "ampere" in str(base.units)

    def test_turn_equals_tau_radians(self, ureg: UnitRegistry) -> None:
        import math

        q = ureg.Quantity(1, "turn").to("radian")
        assert abs(q.magnitude - math.tau) < 1e-10

    def test_degF_to_kelvin(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(32, "degree_Fahrenheit").to("kelvin")
        assert abs(q.magnitude - 273.15) < 1e-4


class TestDimensionDefinitions:
    """Verify dimension definitions through dimensionality checks."""

    def test_length_dimension(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(1, "meter").dimensionality == "[length]"

    def test_time_dimension(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(1, "second").dimensionality == "[time]"

    def test_speed_dimension(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "meter / second")
        dim = q.dimensionality
        assert "[length]" in dim
        assert "[time]" in dim

    def test_incompatible_dimensions(self, ureg: UnitRegistry) -> None:
        with pytest.raises(DimensionalityError):
            ureg.Quantity(1, "meter").to("second")


class TestPrefixDefinitions:
    """Verify prefix definitions work through conversions."""

    PREFIXES = [
        ("kilometer", "meter", 1000.0),
        ("millimeter", "meter", 0.001),
        ("micrometer", "meter", 1e-6),
        ("nanometer", "meter", 1e-9),
        ("megagram", "gram", 1e6),
        ("kilogram", "gram", 1000.0),
    ]

    @pytest.mark.parametrize(
        ("prefixed", "base", "factor"),
        PREFIXES,
        ids=[f"{p}->{b}" for p, b, _ in PREFIXES],
    )
    def test_prefix_conversion(
        self,
        ureg: UnitRegistry,
        prefixed: str,
        base: str,
        factor: float,
    ) -> None:
        q = ureg.Quantity(1, prefixed).to(base)
        assert q.magnitude == pytest.approx(factor, rel=1e-10)


class TestUnitRegistryContains:
    """Verify that defined units can be found in the registry."""

    def test_builtin_unit_exists(self, ureg: UnitRegistry) -> None:
        assert "meter" in ureg
        assert "second" in ureg
        assert "kilogram" in ureg

    def test_unknown_unit_not_found(self, ureg: UnitRegistry) -> None:
        assert "foobar_nonexistent" not in ureg

    def test_defined_unit_exists(self, ureg: UnitRegistry) -> None:
        ureg.define("smoot = 1.7018 * meter")
        assert "smoot" in ureg

    def test_get_name(self, ureg: UnitRegistry) -> None:
        assert ureg.get_name("m") == "meter"

    def test_get_symbol(self, ureg: UnitRegistry) -> None:
        assert ureg.get_symbol("meter") == "m"
