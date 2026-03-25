"""Application registry tests - ported from pint's test_application_registry.py."""

from __future__ import annotations

import pickle

import pytest
from pintrs import (
    Measurement,
    UnitRegistry,
    get_application_registry,
    set_application_registry,
)


@pytest.fixture(autouse=True)
def _reset_app_registry() -> None:  # type: ignore[misc]
    """Reset application registry between tests."""
    import pintrs

    old = pintrs._application_registry
    yield  # type: ignore[misc]
    pintrs._application_registry = old


class TestDefaultApplicationRegistry:
    def test_get_application_registry(self) -> None:
        ureg = get_application_registry()
        assert ureg is not None
        u = ureg.parse_units("kg")
        assert str(u) in {"kg", "kilogram"}

    def test_get_returns_same_instance(self) -> None:
        ureg1 = get_application_registry()
        ureg2 = get_application_registry()
        assert ureg1 is ureg2

    def test_set_application_registry(self) -> None:
        new_ureg = UnitRegistry()
        set_application_registry(new_ureg)
        assert get_application_registry() is new_ureg

    def test_quantity_operations_with_app_registry(self) -> None:
        ureg = get_application_registry()
        q = ureg.Quantity(1, "meter")
        assert q.to("cm").magnitude == pytest.approx(100.0)

    @pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
    def test_pickle_quantity(self, protocol: int) -> None:
        ureg = get_application_registry()
        q = ureg.Quantity(123, "kg")
        restored = pickle.loads(pickle.dumps(q, protocol))
        assert restored.magnitude == pytest.approx(123.0)
        assert str(restored.units) in {"kg", "kilogram"}

    @pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
    def test_pickle_unit(self, protocol: int) -> None:
        ureg = get_application_registry()
        u = ureg.parse_units("kg")
        restored = pickle.loads(pickle.dumps(u, protocol))
        assert str(restored) in {"kg", "kilogram"}


class TestCustomApplicationRegistry:
    def test_custom_unit_in_app_registry(self) -> None:
        ureg = UnitRegistry()
        ureg.define("foo = [test_dim]")
        ureg.define("foo_half = foo / 2")
        set_application_registry(ureg)

        app = get_application_registry()
        q = app.Quantity(1, "foo")
        assert q.to("foo_half").magnitude == pytest.approx(2.0)

    def test_swap_registry(self) -> None:
        ureg1 = UnitRegistry()
        ureg2 = UnitRegistry()

        set_application_registry(ureg1)
        assert get_application_registry() is ureg1

        set_application_registry(ureg2)
        assert get_application_registry() is ureg2


class TestMeasurementWithRegistry:
    def test_measurement_from_registry(self) -> None:
        ureg = get_application_registry()
        q = ureg.Quantity(10, "meter")
        m = Measurement(q, 0.5)
        assert m.value.magnitude == pytest.approx(10.0)
        assert m.error.magnitude == pytest.approx(0.5)

    def test_measurement_str(self) -> None:
        ureg = get_application_registry()
        q = ureg.Quantity(10, "meter")
        m = Measurement(q, 0.5)
        s = str(m)
        assert "10" in s
        assert "0.5" in s


class TestRegistryIsolation:
    """Tests that registries are independent."""

    def test_multiple_registries_independent(self) -> None:
        ureg1 = UnitRegistry()
        ureg2 = UnitRegistry()
        q1 = ureg1.Quantity(1, "meter")
        q2 = ureg2.Quantity(1, "meter")
        assert q1.magnitude == q2.magnitude
        assert str(q1.units) == str(q2.units)

    def test_custom_unit_only_in_own_registry(self) -> None:
        ureg1 = UnitRegistry()
        ureg1.define("wobble = [dim_wobble]")
        q = ureg1.Quantity(1, "wobble")
        assert q.magnitude == 1.0

        ureg2 = UnitRegistry()
        with pytest.raises(Exception):
            ureg2.Quantity(1, "wobble")

    def test_set_restores_previous(self) -> None:
        original = get_application_registry()
        new_ureg = UnitRegistry()
        set_application_registry(new_ureg)
        assert get_application_registry() is new_ureg
        set_application_registry(original)
        assert get_application_registry() is original


class TestRegistryConversions:
    """Tests for quantity operations through the application registry."""

    def test_unit_conversion_kg_to_g(self) -> None:
        ureg = get_application_registry()
        q = ureg.Quantity(1, "kg")
        result = q.to("g")
        assert result.magnitude == pytest.approx(1000.0)

    def test_unit_conversion_m_to_km(self) -> None:
        ureg = get_application_registry()
        q = ureg.Quantity(1500, "meter")
        result = q.to("kilometer")
        assert result.magnitude == pytest.approx(1.5)

    def test_quantity_arithmetic(self) -> None:
        ureg = get_application_registry()
        q1 = ureg.Quantity(1, "meter")
        q2 = ureg.Quantity(2, "meter")
        result = q1 + q2
        assert result.magnitude == pytest.approx(3.0)

    def test_quantity_multiplication(self) -> None:
        ureg = get_application_registry()
        q = ureg.Quantity(3, "meter")
        result = q * 2
        assert result.magnitude == pytest.approx(6.0)

    def test_incompatible_addition_raises(self) -> None:
        ureg = get_application_registry()
        q1 = ureg.Quantity(1, "meter")
        q2 = ureg.Quantity(1, "second")
        with pytest.raises(Exception):
            _ = q1 + q2


class TestSwapApplicationRegistry:
    """Tests for swapping the application registry."""

    def test_swap_preserves_quantities(self) -> None:
        ureg1 = UnitRegistry()
        ureg2 = UnitRegistry()
        set_application_registry(ureg1)
        q1 = get_application_registry().Quantity(10, "meter")
        set_application_registry(ureg2)
        q2 = get_application_registry().Quantity(10, "meter")
        assert q1.magnitude == q2.magnitude

    def test_swap_custom_units(self) -> None:
        ureg1 = UnitRegistry()
        ureg1.define("flurp = [dim_flurp]")
        ureg1.define("flurp_half = flurp / 2")
        set_application_registry(ureg1)
        app = get_application_registry()
        q = app.Quantity(1, "flurp")
        assert q.to("flurp_half").magnitude == pytest.approx(2.0)

    def test_swap_back_to_default(self) -> None:
        default = get_application_registry()
        custom = UnitRegistry()
        set_application_registry(custom)
        assert get_application_registry() is custom
        set_application_registry(default)
        assert get_application_registry() is default


class TestPickleWithCustomRegistry:
    """Pickle tests with custom registries."""

    @pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
    def test_pickle_quantity_custom_registry(self, protocol: int) -> None:
        ureg = UnitRegistry()
        q = ureg.Quantity(42, "meter")
        restored = pickle.loads(pickle.dumps(q, protocol))
        assert restored.magnitude == pytest.approx(42.0)

    def test_pickle_preserves_units(self) -> None:
        ureg = get_application_registry()
        q = ureg.Quantity(3.14, "kilogram")
        restored = pickle.loads(pickle.dumps(q))
        assert str(restored.units) in {"kg", "kilogram"}
        assert restored.magnitude == pytest.approx(3.14)
