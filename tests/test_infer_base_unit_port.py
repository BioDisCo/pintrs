"""Infer base unit and to_compact tests - ported from pint's test_infer_base_unit.py.

Tests to_compact() through the quantity API. infer_base_unit is not
exposed by pintrs, so those tests are adapted to verify equivalent
behavior through conversions and to_base_units().
"""

from __future__ import annotations

import pytest
from pintrs import UnitRegistry


@pytest.fixture
def ureg() -> UnitRegistry:
    return UnitRegistry()


class TestBaseUnitInference:
    """Test that compound prefix units reduce to correct base units."""

    def test_mm_times_nm_reduces_to_m2(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "millimeter * nanometer")
        base = q.to_base_units()
        expected = ureg.Quantity(1e-12, "meter ** 2")
        assert abs(base.magnitude - expected.magnitude) < 1e-22
        assert str(base.units) == str(expected.units)

    def test_units_cancelling_to_seconds(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "meter") * ureg.Quantity(1, "millimeter")
        q = q / ureg.Quantity(1, "meter") / ureg.Quantity(1, "micrometer")
        q = q * ureg.Quantity(1, "second")
        base = q.to_base_units()
        assert base.is_compatible_with("second")

    def test_volt_prefixes(self, ureg: UnitRegistry) -> None:
        r = ureg.Quantity(1, "V") * ureg.Quantity(1, "mV") / ureg.Quantity(1, "kV")
        base = r.to_base_units()
        expected = ureg.Quantity(1, "V").to_base_units()
        assert str(base.units) == str(expected.units)
        assert r.is_compatible_with("V")
        assert abs(r.to("uV").magnitude - 1.0) < 1e-6


class TestToCompact:
    """Test to_compact() method."""

    def test_compact_small(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(0.001, "meter")
        compact = q.to_compact()
        assert abs(compact.magnitude - 1.0) < 1e-10
        assert "milli" in str(compact) or "mm" in str(compact)

    def test_compact_large(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5000, "meter")
        compact = q.to_compact()
        assert abs(compact.magnitude - 5.0) < 1e-10
        assert "km" in str(compact) or "kilo" in str(compact)

    def test_compact_very_large(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1500, "meter")
        compact = q.to_compact()
        assert abs(compact.magnitude - 1.5) < 1e-10

    def test_compact_micro(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(0.000001, "meter")
        compact = q.to_compact()
        assert abs(compact.magnitude - 1.0) < 1e-4

    def test_compact_preserves_value(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1000000000, "meter")
        compact = q.to_compact()
        roundtrip = compact.to("meter")
        assert abs(roundtrip.magnitude - 1000000000) < 1e-3

    def test_compact_compound_units(self, ureg: UnitRegistry) -> None:
        r = (
            ureg.Quantity(1000000000, "m")
            * ureg.Quantity(1, "mm")
            / ureg.Quantity(1, "s")
            / ureg.Quantity(1, "ms")
        )
        compact = r.to_compact()
        expected_in_base = r.to_base_units()
        compact_in_base = compact.to_base_units()
        assert abs(compact_in_base.magnitude - expected_in_base.magnitude) < 1e-3

    def test_compact_already_good(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5, "meter")
        compact = q.to_compact()
        roundtrip = compact.to("meter")
        assert abs(roundtrip.magnitude - 5.0) < 1e-10

    def test_compact_dimensionless(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(42, "dimensionless")
        compact = q.to_compact()
        assert abs(compact.magnitude - 42) < 1e-10


class TestToCompactWithVolts:
    """Test to_compact with voltage units."""

    def test_millivolt_compact(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(0.001, "V")
        compact = q.to_compact()
        assert abs(compact.magnitude - 1.0) < 1e-10

    def test_kilovolt_compact(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1000, "V")
        compact = q.to_compact()
        assert abs(compact.magnitude - 1.0) < 1e-10
