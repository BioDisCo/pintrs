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
