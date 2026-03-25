"""Tests for newly added drop-in compatibility features."""

from __future__ import annotations

import pytest
from pintrs import (
    Context,
    Group,
    Measurement,
    Quantity,
    System,
    UnitRegistry,
    get_application_registry,
    set_application_registry,
)


@pytest.fixture
def ureg() -> UnitRegistry:
    return UnitRegistry()


class TestRegistryMethods:
    def test_get_base_units(self, ureg: UnitRegistry) -> None:
        factor, unit = ureg.get_base_units("kilometer")
        assert abs(factor - 1000) < 1e-10
        assert str(unit) == "m"

    def test_get_root_units(self, ureg: UnitRegistry) -> None:
        factor, _unit = ureg.get_root_units("kilometer")
        assert abs(factor - 1000) < 1e-10

    def test_get_dimensionality(self, ureg: UnitRegistry) -> None:
        assert ureg.get_dimensionality("meter") == "[length]"

    def test_get_name(self, ureg: UnitRegistry) -> None:
        assert ureg.get_name("m") == "meter"

    def test_get_symbol(self, ureg: UnitRegistry) -> None:
        assert ureg.get_symbol("meter") == "m"

    def test_is_compatible_with(self, ureg: UnitRegistry) -> None:
        assert ureg.is_compatible_with("meter", "foot")
        assert not ureg.is_compatible_with("meter", "second")

    def test_parse_unit_name(self, ureg: UnitRegistry) -> None:
        result = ureg.parse_unit_name("kilometer")
        assert len(result) > 0
        assert result[0][0] == "kilo"
        assert result[0][1] == "meter"

    def test_load_definitions(self, ureg: UnitRegistry) -> None:
        ureg.load_definitions("test_unit_xyz = 42 * meter")
        assert "test_unit_xyz" in ureg

    def test_default_format(self, ureg: UnitRegistry) -> None:
        assert ureg.default_format == "D"

    def test_auto_reduce_dimensions(self, ureg: UnitRegistry) -> None:
        assert ureg.auto_reduce_dimensions is False


class TestQuantityMethods:
    def test_to_preferred(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5, "km")
        result = q.to_preferred(["meter"])
        assert abs(result.magnitude - 5000) < 1e-10

    def test_to_preferred_no_match(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5, "km")
        result = q.to_preferred(["second"])
        assert result.magnitude > 0

    def test_ito_preferred(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5, "km")
        q.ito_preferred(["meter"])
        assert abs(q.magnitude - 5000) < 1e-10

    def test_to_unprefixed(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5, "km")
        result = q.to_unprefixed()
        assert abs(result.magnitude - 5000) < 1e-10
        assert str(result.units) == "m"

    def test_ito_unprefixed(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5, "km")
        q.ito_unprefixed()
        assert abs(q.magnitude - 5000) < 1e-10

    def test_ito_reduced_units(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5, "km")
        q.ito_reduced_units()
        assert q.magnitude > 0

    def test_to_tuple(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "meter")
        mag, items = q.to_tuple()
        assert mag == 5.0
        assert len(items) == 1
        assert items[0][0] == "meter"

    def test_from_tuple(self) -> None:
        q = Quantity.from_tuple((3.5, [("meter", 1.0)]))
        assert abs(q.magnitude - 3.5) < 1e-10

    def test_unit_items(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5, "m/s")
        items = q.unit_items()
        assert len(items) == 2

    def test_compare(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(1, "km")
        b = ureg.Quantity(500, "meter")
        assert a.compare(b, ">")
        assert not a.compare(b, "==")
        assert a.compare(b, ">=")

    def test_to_timedelta(self, ureg: UnitRegistry) -> None:
        import datetime

        q = ureg.Quantity(90, "second")
        td = q.to_timedelta()
        assert isinstance(td, datetime.timedelta)
        assert td.total_seconds() == 90.0

    def test_plus_minus(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "meter")
        m = q.plus_minus(0.1)
        assert isinstance(m, Measurement)
        assert "5.0" in str(m)


class TestUnitMethods:
    def test_from_(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("meter")
        q = u.from_(ureg.Quantity(1, "km"))
        assert abs(q.magnitude - 1000) < 1e-10

    def test_m_from(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("meter")
        assert abs(u.m_from(ureg.Quantity(1, "km")) - 1000) < 1e-10

    def test_systems(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("meter")
        assert isinstance(u.systems, list)

    def test_compare(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("meter")
        assert u.compare(u, "==")


class TestApplicationRegistry:
    def test_set_and_get(self) -> None:
        ureg = UnitRegistry()
        set_application_registry(ureg)
        assert get_application_registry() is ureg

    def test_auto_create(self) -> None:
        import pintrs

        pintrs._application_registry = None
        reg = get_application_registry()
        assert reg is not None


class TestContextStub:
    def test_create(self) -> None:
        ctx = Context("spectroscopy")
        assert ctx.name == "spectroscopy"

    def test_add_transformation(self) -> None:
        ctx = Context("test")
        ctx.add_transformation("[length]", "[frequency]", lambda x: 1 / x)
        assert ("[length]", "[frequency]") in ctx._transforms


class TestGroupStub:
    def test_create(self) -> None:
        g = Group("length")
        g.add_units("meter", "foot", "inch")
        assert "meter" in g.members


class TestSystemStub:
    def test_create(self) -> None:
        s = System("mks")
        assert s.name == "mks"
        assert isinstance(s.base_units, dict)


class TestMeasurementArithmetic:
    def test_add(self, ureg: UnitRegistry) -> None:
        m1 = Measurement(ureg.Quantity(5.0, "meter"), 0.1)
        m2 = Measurement(ureg.Quantity(3.0, "meter"), 0.2)
        result = m1 + m2
        assert abs(result.magnitude - 8.0) < 1e-10

    def test_sub(self, ureg: UnitRegistry) -> None:
        m1 = Measurement(ureg.Quantity(5.0, "meter"), 0.1)
        m2 = Measurement(ureg.Quantity(3.0, "meter"), 0.2)
        result = m1 - m2
        assert abs(result.magnitude - 2.0) < 1e-10

    def test_mul_scalar(self, ureg: UnitRegistry) -> None:
        m = Measurement(ureg.Quantity(5.0, "meter"), 0.1)
        result = m * 2
        assert abs(result.magnitude - 10.0) < 1e-10

    def test_div_scalar(self, ureg: UnitRegistry) -> None:
        m = Measurement(ureg.Quantity(10.0, "meter"), 0.2)
        result = m / 2
        assert abs(result.magnitude - 5.0) < 1e-10

    def test_error_with_quantity(self, ureg: UnitRegistry) -> None:
        m = Measurement(ureg.Quantity(5.0, "meter"), ureg.Quantity(10, "centimeter"))
        assert abs(m.error.magnitude - 0.1) < 1e-10
