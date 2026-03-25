"""Context tests - ported from pint's test_contexts.py."""

from __future__ import annotations

import pytest
from pintrs import DimensionalityError, UnitRegistry
from pintrs.context import Context


@pytest.fixture
def ureg() -> UnitRegistry:
    return UnitRegistry()


class TestContextCreation:
    def test_create_context(self) -> None:
        ctx = Context("test_ctx_create")
        assert ctx.name == "test_ctx_create"

    def test_context_with_transform(self) -> None:
        ctx = Context("test_ctx_transform")
        ctx.add_transformation(
            "[length]", "[time]", lambda ureg, x: x / ureg.Quantity(3e8, "m/s")
        )
        assert len(ctx.transforms) == 1

    def test_context_registry(self) -> None:
        name = "test_ctx_registry_unique"
        ctx = Context(name)
        assert Context.get(name) is ctx

    def test_context_get_nonexistent(self) -> None:
        assert Context.get("nonexistent_context_xyz") is None

    def test_context_defaults(self) -> None:
        ctx = Context("test_ctx_defaults", defaults={"n": 1.0})
        assert ctx.defaults["n"] == 1.0


class TestBuiltinContexts:
    def test_spectroscopy_context_exists(self) -> None:
        ctx = Context.get("spectroscopy")
        assert ctx is not None
        assert ctx.name == "spectroscopy"

    def test_boltzmann_context_exists(self) -> None:
        ctx = Context.get("Boltzmann")
        assert ctx is not None

    def test_chemistry_context_exists(self) -> None:
        ctx = Context.get("chemistry")
        assert ctx is not None


class TestContextConversions:
    def test_spectroscopy_wavelength_to_frequency(self, ureg: UnitRegistry) -> None:
        with ureg.context("spectroscopy"):
            wl = ureg.Quantity(500, "nanometer")
            freq = wl.to("hertz")
            assert freq.magnitude > 0

    def test_spectroscopy_frequency_to_wavelength(self, ureg: UnitRegistry) -> None:
        with ureg.context("spectroscopy"):
            freq = ureg.Quantity(6e14, "hertz")
            wl = freq.to("nanometer")
            assert wl.magnitude > 0

    def test_spectroscopy_energy_to_wavelength(self, ureg: UnitRegistry) -> None:
        with ureg.context("spectroscopy"):
            energy = ureg.Quantity(2.5, "eV")
            wl = energy.to("nanometer")
            assert wl.magnitude > 0

    def test_spectroscopy_wavelength_to_energy(self, ureg: UnitRegistry) -> None:
        with ureg.context("spectroscopy"):
            wl = ureg.Quantity(500, "nanometer")
            energy = wl.to("eV")
            assert energy.magnitude > 0

    def test_spectroscopy_round_trip(self, ureg: UnitRegistry) -> None:
        with ureg.context("spectroscopy"):
            wl = ureg.Quantity(500, "nanometer")
            freq = wl.to("hertz")
            wl_back = freq.to("nanometer")
            assert wl_back.magnitude == pytest.approx(500, rel=1e-6)

    def test_without_context_raises(self, ureg: UnitRegistry) -> None:
        wl = ureg.Quantity(500, "nanometer")
        with pytest.raises(DimensionalityError):
            wl.to("hertz")

    def test_boltzmann_energy_to_temperature(self, ureg: UnitRegistry) -> None:
        with ureg.context("Boltzmann"):
            energy = ureg.Quantity(1, "eV")
            temp = energy.to("kelvin")
            assert temp.magnitude > 0

    def test_boltzmann_temperature_to_energy(self, ureg: UnitRegistry) -> None:
        with ureg.context("Boltzmann"):
            temp = ureg.Quantity(300, "kelvin")
            energy = temp.to("eV")
            assert energy.magnitude > 0

    def test_boltzmann_round_trip(self, ureg: UnitRegistry) -> None:
        with ureg.context("Boltzmann"):
            temp = ureg.Quantity(300, "kelvin")
            energy = temp.to("joule")
            temp_back = energy.to("kelvin")
            assert temp_back.magnitude == pytest.approx(300, rel=1e-6)


class TestContextManager:
    def test_context_manager_enables_and_disables(self, ureg: UnitRegistry) -> None:
        wl = ureg.Quantity(500, "nanometer")
        with pytest.raises(DimensionalityError):
            wl.to("hertz")

        with ureg.context("spectroscopy"):
            freq = wl.to("hertz")
            assert freq.magnitude > 0

        with pytest.raises(DimensionalityError):
            wl.to("hertz")

    def test_nested_contexts(self, ureg: UnitRegistry) -> None:
        with ureg.context("spectroscopy"), ureg.context("Boltzmann"):
            wl = ureg.Quantity(500, "nanometer")
            freq = wl.to("hertz")
            assert freq.magnitude > 0
            temp = ureg.Quantity(300, "kelvin")
            energy = temp.to("eV")
            assert energy.magnitude > 0


class TestContextEnableDisable:
    def test_enable_contexts(self, ureg: UnitRegistry) -> None:
        ureg.enable_contexts("spectroscopy")
        wl = ureg.Quantity(500, "nanometer")
        freq = wl.to("hertz")
        assert freq.magnitude > 0
        ureg.disable_contexts("spectroscopy")

    def test_disable_contexts(self, ureg: UnitRegistry) -> None:
        ureg.enable_contexts("spectroscopy")
        ureg.disable_contexts("spectroscopy")
        wl = ureg.Quantity(500, "nanometer")
        with pytest.raises(DimensionalityError):
            wl.to("hertz")


class TestCustomContext:
    def test_add_custom_context(self, ureg: UnitRegistry) -> None:
        ctx = Context("custom_port_test")
        ctx.add_transformation(
            "[length]",
            "[time]",
            lambda ureg, x: x / ureg.Quantity(1, "m/s"),
        )
        ureg.add_context(ctx)
        with ureg.context("custom_port_test"):
            q = ureg.Quantity(10, "meter")
            t = q.to("second")
            assert t.magnitude == pytest.approx(10.0)

    def test_remove_context(self, ureg: UnitRegistry) -> None:
        ctx = Context("removable_port_test")
        ureg.add_context(ctx)
        assert Context.get("removable_port_test") is not None
        ureg.remove_context("removable_port_test")
        assert Context.get("removable_port_test") is None

    def test_bidirectional_transform(self, ureg: UnitRegistry) -> None:
        ctx = Context("bidir_port_test")
        ctx.add_transformation(
            "[length]",
            "[time]",
            lambda ureg, x: x / ureg.Quantity(1, "m/s"),
        )
        ctx.add_transformation(
            "[time]",
            "[length]",
            lambda ureg, x: x * ureg.Quantity(1, "m/s"),
        )
        ureg.add_context(ctx)
        with ureg.context("bidir_port_test"):
            length = ureg.Quantity(10, "meter")
            time = length.to("second")
            length_back = time.to("meter")
            assert length_back.magnitude == pytest.approx(10.0)
