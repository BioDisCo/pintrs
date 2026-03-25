"""Tests for context-based conversions."""

from __future__ import annotations

import pytest
from pintrs import Context, DimensionalityError, UnitRegistry


@pytest.fixture
def ureg() -> UnitRegistry:
    return UnitRegistry()


class TestContextCreation:
    def test_create_named(self) -> None:
        ctx = Context("test_ctx")
        assert ctx.name == "test_ctx"

    def test_add_transformation(self) -> None:
        ctx = Context("test")
        ctx.add_transformation("[length]", "[time]", lambda _ureg, v: v / 3e8)
        assert ("[length]", "[time]") in ctx._transforms

    def test_builtin_spectroscopy_exists(self) -> None:
        ctx = Context.get("spectroscopy")
        assert ctx is not None
        assert ctx.name == "spectroscopy"

    def test_builtin_Boltzmann_exists(self) -> None:
        ctx = Context.get("Boltzmann")
        assert ctx is not None

    def test_builtin_chemistry_exists(self) -> None:
        ctx = Context.get("chemistry")
        assert ctx is not None

    def test_from_lines(self) -> None:
        ctx = Context.from_lines(["my_context"])
        assert ctx.name == "my_context"


class TestSpectroscopyContext:
    def test_wavelength_to_frequency(self, ureg: UnitRegistry) -> None:
        with ureg.context("spectroscopy"):
            wavelength = ureg.Quantity(500, "nanometer")
            freq = wavelength.to("hertz")
            # c / lambda = 3e8 / 500e-9 = 6e14 Hz
            assert abs(freq.magnitude - 5.99585e14) < 1e11

    def test_frequency_to_wavelength(self, ureg: UnitRegistry) -> None:
        with ureg.context("spectroscopy"):
            freq = ureg.Quantity(6e14, "hertz")
            wl = freq.to("nanometer")
            assert abs(wl.magnitude - 499.65) < 1

    def test_wavelength_to_energy(self, ureg: UnitRegistry) -> None:
        with ureg.context("spectroscopy"):
            wl = ureg.Quantity(500, "nanometer")
            energy = wl.to("joule")
            # E = hc/lambda ~ 3.97e-19 J
            assert abs(energy.magnitude - 3.97e-19) < 1e-21

    def test_energy_to_frequency(self, ureg: UnitRegistry) -> None:
        with ureg.context("spectroscopy"):
            energy = ureg.Quantity(1.0, "electron_volt")
            freq = energy.to("hertz")
            # 1 eV ~ 2.418e14 Hz
            assert abs(freq.magnitude - 2.418e14) < 1e12

    def test_no_context_raises(self, ureg: UnitRegistry) -> None:
        wl = ureg.Quantity(500, "nanometer")
        with pytest.raises(DimensionalityError):
            wl.to("hertz")

    def test_context_exits_cleanly(self, ureg: UnitRegistry) -> None:
        with ureg.context("spectroscopy"):
            ureg.Quantity(500, "nanometer").to("hertz")
        with pytest.raises(DimensionalityError):
            ureg.Quantity(500, "nanometer").to("hertz")


class TestCustomContext:
    def test_user_defined_context(self, ureg: UnitRegistry) -> None:
        ctx = Context("my_ctx")
        ctx.add_transformation(
            "[length]",
            "[time]",
            lambda _ureg, v, **_kw: v / 3e8,
        )
        ureg.add_context(ctx)
        with ureg.context("my_ctx"):
            q = ureg.Quantity(3e8, "meter")
            result = q.to("second")
            assert abs(result.magnitude - 1.0) < 1e-6


class TestContextEnableDisable:
    def test_enable_disable(self, ureg: UnitRegistry) -> None:
        ureg.enable_contexts("spectroscopy")
        ureg.disable_contexts("spectroscopy")
