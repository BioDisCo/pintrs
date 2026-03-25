"""Context-based conversions for pintrs.

Contexts allow converting between otherwise incompatible units by defining
custom transformation functions. For example, the spectroscopy context
allows converting between wavelength, frequency, and energy.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from pintrs._core import Quantity as ScalarQuantity
    from pintrs._core import UnitRegistry as _UnitRegistry

_PLANCK = 6.62607015e-34
_SPEED_OF_LIGHT = 299792458.0
_K_BOLTZMANN = 1.380649e-23
_AVOGADRO = 6.02214076e23

_DIM_LENGTH = "[length]"
_DIM_FREQUENCY = "1 / [time]"
_DIM_ENERGY = "[length] ** 2 * [mass] / [time] ** 2"


class Context:
    """A context that enables additional unit conversions.

    Contexts allow converting between otherwise incompatible dimensions
    by registering transformation functions between dimension pairs.

    Args:
        name: Name of the context.
        defaults: Default parameter values for transformations.
    """

    _REGISTRY: ClassVar[dict[str, Context]] = {}

    _DIM_TO_SI_UNIT: ClassVar[dict[str, str]] = {
        "[length]": "meter",
        "[time]": "second",
        "[mass]": "kilogram",
        "[temperature]": "kelvin",
        "[current]": "ampere",
        "[substance]": "mole",
        "[luminosity]": "candela",
        "1 / [time]": "hertz",
        "[length] ** 2 * [mass] / [time] ** 2": "joule",
    }

    def __init__(
        self,
        name: str = "",
        defaults: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.name = name
        self._transforms: dict[tuple[str, str], Callable[..., Any]] = {}
        self._defaults: dict[str, Any] = dict(defaults or {})
        self._defaults.update(kwargs)
        if name:
            Context._REGISTRY[name] = self

    @property
    def transforms(self) -> dict[tuple[str, str], Callable[..., Any]]:
        """Registered transformations."""
        return dict(self._transforms)

    @property
    def defaults(self) -> dict[str, Any]:
        """Default parameters passed to transformations."""
        return dict(self._defaults)

    def add_transformation(
        self,
        src: str,
        dst: str,
        func: Callable[..., Any],
    ) -> None:
        """Register a transformation function between two dimensions.

        Args:
            src: Source dimensionality string (e.g. "[length]").
            dst: Destination dimensionality string (e.g. "1 / [time]").
            func: Callable(ureg, value, **kwargs) -> transformed value.
        """
        self._transforms[(src, dst)] = func

    def transform(
        self,
        ureg: _UnitRegistry,
        quantity: ScalarQuantity,
        src_dim: str,
        dst_dim: str,
    ) -> Any:
        """Apply a registered transformation.

        Returns:
            Transformed magnitude in SI base units of dst, or None if no
            transformation is registered for this pair.
        """
        func = self._transforms.get((src_dim, dst_dim))
        if func is None:
            return None
        result = func(ureg, quantity, **self._defaults)
        if hasattr(result, "check") and result.check(dst_dim):
            return result

        si_unit = self._DIM_TO_SI_UNIT.get(src_dim)
        if si_unit is not None:
            mag = float(quantity.m_as(si_unit))
        else:
            base = quantity.to_base_units()
            mag = float(base.magnitude)
        return func(ureg, mag, **self._defaults)

    @staticmethod
    def from_lines(
        lines: list[str],
        to_base_func: Any = None,  # noqa: ARG004
    ) -> Context:
        """Create a Context from definition lines."""
        name = lines[0].strip() if lines else ""
        return Context(name)

    @staticmethod
    def get(name: str) -> Context | None:
        """Look up a context by name."""
        return Context._REGISTRY.get(name)


def _build_spectroscopy_context() -> Context:
    """Build the spectroscopy context (wavelength <-> frequency <-> energy)."""
    ctx = Context("spectroscopy")

    def _len_to_freq(_ureg: _UnitRegistry, value: ScalarQuantity, **_kw: Any) -> Any:
        return _ureg.Quantity(_SPEED_OF_LIGHT, "meter / second") / value

    def _freq_to_len(_ureg: _UnitRegistry, value: ScalarQuantity, **_kw: Any) -> Any:
        return _ureg.Quantity(_SPEED_OF_LIGHT, "meter / second") / value

    def _len_to_energy(_ureg: _UnitRegistry, value: ScalarQuantity, **_kw: Any) -> Any:
        hc = _ureg.Quantity(_PLANCK * _SPEED_OF_LIGHT, "joule * meter")
        return hc / value

    def _energy_to_len(_ureg: _UnitRegistry, value: ScalarQuantity, **_kw: Any) -> Any:
        hc = _ureg.Quantity(_PLANCK * _SPEED_OF_LIGHT, "joule * meter")
        return hc / value

    def _freq_to_energy(_ureg: _UnitRegistry, value: ScalarQuantity, **_kw: Any) -> Any:
        return _ureg.Quantity(_PLANCK, "joule / hertz") * value

    def _energy_to_freq(_ureg: _UnitRegistry, value: ScalarQuantity, **_kw: Any) -> Any:
        return value / _ureg.Quantity(_PLANCK, "joule / hertz")

    ctx.add_transformation(_DIM_LENGTH, _DIM_FREQUENCY, _len_to_freq)
    ctx.add_transformation(_DIM_FREQUENCY, _DIM_LENGTH, _freq_to_len)
    ctx.add_transformation(_DIM_LENGTH, _DIM_ENERGY, _len_to_energy)
    ctx.add_transformation(_DIM_ENERGY, _DIM_LENGTH, _energy_to_len)
    ctx.add_transformation(_DIM_FREQUENCY, _DIM_ENERGY, _freq_to_energy)
    ctx.add_transformation(_DIM_ENERGY, _DIM_FREQUENCY, _energy_to_freq)

    return ctx


def _build_boltzmann_context() -> Context:
    """Build the Boltzmann context (energy <-> temperature)."""
    ctx = Context("Boltzmann")

    def _energy_to_temp(_ureg: _UnitRegistry, value: ScalarQuantity, **_kw: Any) -> Any:
        return value / _ureg.Quantity(_K_BOLTZMANN, "joule / kelvin")

    def _temp_to_energy(_ureg: _UnitRegistry, value: ScalarQuantity, **_kw: Any) -> Any:
        return value * _ureg.Quantity(_K_BOLTZMANN, "joule / kelvin")

    ctx.add_transformation(_DIM_ENERGY, "[temperature]", _energy_to_temp)
    ctx.add_transformation("[temperature]", _DIM_ENERGY, _temp_to_energy)

    return ctx


def _build_chemistry_context() -> Context:
    """Build the chemistry context (amount <-> count via Avogadro)."""
    ctx = Context("chemistry")

    def _amount_to_count(
        _ureg: _UnitRegistry,
        value: ScalarQuantity,
        **_kw: Any,
    ) -> Any:
        return value * _AVOGADRO

    def _count_to_amount(
        _ureg: _UnitRegistry,
        value: ScalarQuantity,
        **_kw: Any,
    ) -> Any:
        return value / _AVOGADRO

    ctx.add_transformation("[substance]", "[dimensionless]", _amount_to_count)
    ctx.add_transformation("[dimensionless]", "[substance]", _count_to_amount)

    return ctx


# Build built-in contexts at import time
_build_spectroscopy_context()
_build_boltzmann_context()
_build_chemistry_context()


class ActiveContexts:
    """Manages which contexts are currently active on a registry."""

    def __init__(self) -> None:
        self._active: list[Context] = []

    @property
    def active(self) -> list[Context]:
        """Currently active contexts."""
        return list(self._active)

    def enable(self, *names: str | Context) -> None:
        """Enable one or more contexts by name or instance."""
        for name in names:
            if isinstance(name, Context):
                ctx: Context = name
            else:
                found = Context.get(name)
                if found is None:
                    msg = f"Unknown context: {name}"
                    raise KeyError(msg)
                ctx = found
            if ctx not in self._active:
                self._active.append(ctx)

    def disable(self, *names: str | Context) -> None:
        """Disable one or more contexts."""
        for name in names:
            if isinstance(name, Context):
                ctx: Context = name
            else:
                found = Context.get(name)
                if found is None:
                    continue
                ctx = found
            if ctx in self._active:
                self._active.remove(ctx)

    def clear(self) -> None:
        """Disable all contexts."""
        self._active.clear()

    @contextlib.contextmanager
    def __call__(
        self,
        *names: str | Context,
    ) -> Generator[None, None, None]:
        """Context manager that temporarily enables contexts."""
        prev = list(self._active)
        self.enable(*names)
        try:
            yield
        finally:
            self._active = prev

    def find_transform(
        self,
        ureg: _UnitRegistry,
        quantity: ScalarQuantity,
        src_dim: str,
        dst_dim: str,
    ) -> Any:
        """Search active contexts for a valid transformation."""
        for ctx in self._active:
            result = ctx.transform(ureg, quantity, src_dim, dst_dim)
            if result is not None:
                return result
        return None
