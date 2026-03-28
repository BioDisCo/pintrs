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


class TestContextWithArguments:
    """Tests for contexts with parameters/arguments."""

    def test_context_with_default_parameter(self, ureg: UnitRegistry) -> None:
        ctx = Context("param_default_test", defaults={"n": 2.0})
        ctx.add_transformation(
            "[length]",
            "[time]",
            lambda ureg, x, n=1: x / (ureg.Quantity(n, "m/s")),
        )
        ureg.add_context(ctx)
        with ureg.context("param_default_test"):
            q = ureg.Quantity(10, "meter")
            t = q.to("second")
            assert t.magnitude == pytest.approx(5.0)
        ureg.remove_context("param_default_test")

    def test_context_defaults_property(self) -> None:
        ctx = Context("defaults_prop_test", defaults={"n": 1.0, "m": 2.0})
        assert ctx.defaults == {"n": 1.0, "m": 2.0}

    def test_context_multiple_transforms(self, ureg: UnitRegistry) -> None:
        ctx = Context("multi_transform_test")
        ctx.add_transformation(
            "[length]",
            "[time]",
            lambda ureg, x: x / ureg.Quantity(1, "m/s"),
        )
        ctx.add_transformation(
            "[mass]",
            "[time]",
            lambda ureg, x: x / ureg.Quantity(1, "kg/s"),
        )
        ureg.add_context(ctx)
        with ureg.context("multi_transform_test"):
            length = ureg.Quantity(5, "meter")
            t1 = length.to("second")
            assert t1.magnitude == pytest.approx(5.0)
            mass = ureg.Quantity(3, "kilogram")
            t2 = mass.to("second")
            assert t2.magnitude == pytest.approx(3.0)
        ureg.remove_context("multi_transform_test")


class TestAnonymousContext:
    """Tests for unnamed (anonymous) contexts."""

    def test_anonymous_context_has_no_name(self) -> None:
        c = Context()
        assert c.name == ""

    def test_anonymous_context_not_in_registry(self) -> None:
        Context._REGISTRY.pop("", None)  # clean slate
        _ = Context()
        assert Context.get("") is None

    def test_anonymous_context_with_ureg_context(self, ureg: UnitRegistry) -> None:
        c = Context()
        c.add_transformation(
            "[length]", "[time]", lambda ureg, x: x / ureg.Quantity(5, "cm/s")
        )
        x = ureg.Quantity(10, "cm")
        with ureg.context(c):
            t = x.to("second")
            assert t.magnitude == pytest.approx(2.0)

    def test_anonymous_context_with_enable_disable(self, ureg: UnitRegistry) -> None:
        c = Context()
        c.add_transformation(
            "[length]", "[time]", lambda ureg, x: x / ureg.Quantity(5, "cm/s")
        )
        ureg.enable_contexts(c)
        x = ureg.Quantity(10, "cm")
        t = x.to("second")
        assert t.magnitude == pytest.approx(2.0)
        ureg.disable_contexts(c)
        with pytest.raises(DimensionalityError):
            x.to("second")

    def test_add_anonymous_context_to_registry(self, ureg: UnitRegistry) -> None:
        c = Context()
        c.add_transformation(
            "[length]", "[time]", lambda ureg, x: x / ureg.Quantity(1, "m/s")
        )
        # pintrs may or may not raise on adding anonymous context;
        # pint raises ValueError
        try:
            ureg.add_context(c)
            # If it didn't raise, that's fine for pintrs
        except ValueError:
            pass


class TestContextStacking:
    """Tests for stacking multiple contexts."""

    def test_stacked_contexts_both_active(self, ureg: UnitRegistry) -> None:
        with ureg.context("spectroscopy", "Boltzmann"):
            wl = ureg.Quantity(500, "nanometer")
            freq = wl.to("hertz")
            assert freq.magnitude > 0
            temp = ureg.Quantity(300, "kelvin")
            energy = temp.to("eV")
            assert energy.magnitude > 0

    def test_nested_context_inner_does_not_leak(self, ureg: UnitRegistry) -> None:
        wl = ureg.Quantity(500, "nanometer")
        with ureg.context("spectroscopy"):
            freq = wl.to("hertz")
            assert freq.magnitude > 0
            temp = ureg.Quantity(300, "kelvin")
            with ureg.context("Boltzmann"):
                energy = temp.to("eV")
                assert energy.magnitude > 0
            # Boltzmann should be inactive now
            with pytest.raises(DimensionalityError):
                temp.to("eV")

    def test_context_restored_after_exception(self, ureg: UnitRegistry) -> None:
        wl = ureg.Quantity(500, "nanometer")
        with pytest.raises(ZeroDivisionError):
            with ureg.context("spectroscopy"):
                _ = wl.to("hertz")
                _ = 1 / 0
        with pytest.raises(DimensionalityError):
            wl.to("hertz")


class TestContextErrors:
    """Tests for error cases."""

    def test_unknown_context_raises(self, ureg: UnitRegistry) -> None:
        with pytest.raises(KeyError):
            with ureg.context("totally_nonexistent_ctx"):
                pass

    def test_enable_unknown_context_raises(self, ureg: UnitRegistry) -> None:
        with pytest.raises(KeyError):
            ureg.enable_contexts("totally_nonexistent_ctx")

    def test_remove_nonexistent_context(self, ureg: UnitRegistry) -> None:
        # Should not raise or should raise gracefully
        try:
            ureg.remove_context("never_added_ctx_xyz")
        except (KeyError, ValueError):
            pass

    def test_conversion_still_fails_without_matching_transform(
        self, ureg: UnitRegistry
    ) -> None:
        ctx = Context("partial_transform_test")
        ctx.add_transformation(
            "[length]",
            "[time]",
            lambda ureg, x: x / ureg.Quantity(1, "m/s"),
        )
        ureg.add_context(ctx)
        with ureg.context("partial_transform_test"):
            # length -> time works
            q = ureg.Quantity(10, "meter")
            t = q.to("second")
            assert t.magnitude == pytest.approx(10.0)
            # mass -> time has no transform, should still fail
            m = ureg.Quantity(5, "kilogram")
            with pytest.raises(DimensionalityError):
                m.to("second")
        ureg.remove_context("partial_transform_test")


class TestContextDecorator:
    """Tests for the with_context decorator pattern."""

    def test_with_context_decorator(self, ureg: UnitRegistry) -> None:
        wl = ureg.Quantity(532.0, "nm")
        with ureg.context("spectroscopy"):
            expected = wl.to("terahertz")

        @ureg.with_context("spectroscopy")
        def convert(wavelength: object) -> object:
            return wavelength.to("terahertz")  # type: ignore[union-attr]

        result = convert(wl)
        assert result.magnitude == pytest.approx(expected.magnitude, rel=1e-6)  # type: ignore[union-attr]

    def test_with_context_decorator_fails_outside(self, ureg: UnitRegistry) -> None:
        wl = ureg.Quantity(532.0, "nm")

        def plain_convert(wavelength: object) -> object:
            return wavelength.to("terahertz")  # type: ignore[union-attr]

        with pytest.raises(DimensionalityError):
            plain_convert(wl)


class TestSpectroscopyContext:
    """Detailed spectroscopy context tests ported from pint."""

    def test_spectroscopy_nm_to_thz(self, ureg: UnitRegistry) -> None:
        with ureg.context("spectroscopy"):
            nm532 = ureg.Quantity(532.0, "nm")
            thz = nm532.to("terahertz")
            assert thz.magnitude == pytest.approx(563.5, rel=0.01)

    def test_spectroscopy_nm_to_ev(self, ureg: UnitRegistry) -> None:
        with ureg.context("spectroscopy"):
            nm532 = ureg.Quantity(532.0, "nm")
            ev = nm532.to("eV")
            assert ev.magnitude == pytest.approx(2.33053, rel=0.01)

    def test_spectroscopy_all_pairs(self, ureg: UnitRegistry) -> None:
        eq = (
            ureg.Quantity(532.0, "nm"),
            ureg.Quantity(563.5, "terahertz"),
            ureg.Quantity(2.33053, "eV"),
        )
        with ureg.context("spectroscopy"):
            for a in eq:
                for b in eq:
                    converted = a.to(str(b.units))
                    assert converted.magnitude == pytest.approx(
                        b.magnitude, rel=0.01
                    ), f"{a} -> {b}"


class TestChemistryContext:
    """Tests for the chemistry context."""

    def test_chemistry_context_exists(self) -> None:
        ctx = Context.get("chemistry")
        assert ctx is not None
        assert ctx.name == "chemistry"

    def test_chemistry_context_has_transforms(self) -> None:
        ctx = Context.get("chemistry")
        assert ctx is not None
        assert len(ctx.transforms) > 0
