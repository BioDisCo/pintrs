"""pintrs - Fast Rust-based Python extension for physical unit manipulation."""

from __future__ import annotations

import contextlib
import functools
from typing import TYPE_CHECKING

from pintrs._core import (
    DefinitionSyntaxError,
    DimensionalityError,
    OffsetUnitCalculusError,
    PintError,
    Quantity,
    RedefinitionError,
    UndefinedUnitError,
    Unit,
    UnitRegistry,
)
from pintrs.numpy_support import ArrayQuantity

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

__all__ = [
    "ArrayQuantity",
    "Context",
    "DefinitionSyntaxError",
    "DimensionalityError",
    "Group",
    "Measurement",
    "OffsetUnitCalculusError",
    "PintError",
    "Quantity",
    "RedefinitionError",
    "System",
    "UndefinedUnitError",
    "Unit",
    "UnitRegistry",
    "check",
    "get_application_registry",
    "make_quantity",
    "set_application_registry",
    "wraps",
]
from importlib.metadata import version as _metadata_version

__version__ = _metadata_version("pintrs")

# --- Application registry singleton ---

_application_registry: UnitRegistry | None = None


def set_application_registry(ureg: UnitRegistry) -> None:
    """Set the application-wide default registry."""
    global _application_registry  # noqa: PLW0603
    _application_registry = ureg


def get_application_registry() -> UnitRegistry:
    """Get the application-wide default registry, creating one if needed."""
    global _application_registry  # noqa: PLW0603
    if _application_registry is None:
        _application_registry = UnitRegistry()
    return _application_registry


# --- Decorators ---


def wraps(
    ureg: UnitRegistry,
    ret: str | Unit | None,
    args: tuple[str | Unit | None, ...] | None = None,
    strict: bool = True,  # noqa: ARG001
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to wrap a function for automatic unit conversion.

    Args:
        ureg: The unit registry to use.
        ret: The return unit (or None for dimensionless).
        args: Tuple of expected argument units (or None for no conversion).
        strict: If True, raise on incompatible units. Currently ignored.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*values: Any) -> Any:
            new_args: list[Any] = []
            if args is not None:
                for val, unit_spec in zip(values, args, strict=False):
                    if unit_spec is None:
                        if isinstance(val, Quantity):
                            new_args.append(val.magnitude)
                        else:
                            new_args.append(val)
                    else:
                        unit_str = str(unit_spec)
                        if isinstance(val, Quantity):
                            new_args.append(val.m_as(unit_str))
                        else:
                            new_args.append(val)
                new_args.extend(values[len(args) :])
            else:
                new_args = list(values)

            result = func(*new_args)

            if ret is not None:
                return ureg.Quantity(result, str(ret))
            return result

        return wrapper

    return decorator


def check(
    ureg: UnitRegistry,  # noqa: ARG001
    *args: str | None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to check argument dimensions before calling.

    Args:
        ureg: The unit registry to use.
        args: Expected dimensionality strings for each argument (or None to skip).
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*values: Any) -> Any:
            for val, expected_dim in zip(values, args, strict=False):
                if expected_dim is None:
                    continue
                if not isinstance(val, Quantity):
                    msg = f"Expected Quantity, got {type(val).__name__}"
                    raise DimensionalityError(msg)
                if not val.check(expected_dim):
                    msg = (
                        f"Argument has dimensionality {val.dimensionality}, "
                        f"expected {expected_dim}"
                    )
                    raise DimensionalityError(msg)
            return func(*values)

        return wrapper

    return decorator


# --- Measurement ---


class Measurement:
    """A quantity with an associated uncertainty.

    This provides basic pint.Measurement compatibility.
    """

    def __init__(
        self,
        value: Quantity | float,
        error: Quantity | float,
        units: str | None = None,
    ) -> None:
        if isinstance(value, Quantity):
            self._value = value
        elif units is not None:
            self._value = Quantity(float(value), units)
        else:
            self._value = Quantity(float(value), "dimensionless")

        if isinstance(error, Quantity):
            self._error = abs(error.to(str(self._value.units)).magnitude)
        else:
            self._error = abs(float(error))

    @property
    def value(self) -> Quantity:
        """The central value."""
        return self._value

    @property
    def error(self) -> Quantity:
        """The uncertainty as a Quantity."""
        return Quantity(self._error, str(self._value.units))

    @property
    def magnitude(self) -> float:
        """The magnitude of the central value."""
        return self._value.magnitude

    @property
    def units(self) -> Unit:
        """The units."""
        return self._value.units

    @property
    def rel(self) -> float:
        """Relative error."""
        if self._value.magnitude == 0:
            return float("inf")
        return abs(self._error / self._value.magnitude)

    def __repr__(self) -> str:
        mag = self._value.magnitude
        units = self._value.units
        return f"<Measurement({mag}, {self._error}, '{units}')>"

    def __str__(self) -> str:
        return f"{self._value.magnitude} +/- {self._error} {self._value.units}"

    def __add__(self, other: Measurement) -> Measurement:
        if isinstance(other, Measurement):
            new_val = self._value + other._value
            new_err = (self._error**2 + other._error**2) ** 0.5
            return Measurement(new_val, new_err)
        return NotImplemented

    def __sub__(self, other: Measurement) -> Measurement:
        if isinstance(other, Measurement):
            new_val = self._value - other._value
            new_err = (self._error**2 + other._error**2) ** 0.5
            return Measurement(new_val, new_err)
        return NotImplemented

    def __mul__(self, other: Measurement | float) -> Measurement:
        if isinstance(other, Measurement):
            new_val = self._value * other._value
            rel_err = (self.rel**2 + other.rel**2) ** 0.5
            new_err = abs(new_val.magnitude) * rel_err
            return Measurement(new_val, new_err)
        if isinstance(other, int | float):
            return Measurement(self._value * other, self._error * abs(other))
        return NotImplemented

    def __truediv__(self, other: Measurement | float) -> Measurement:
        if isinstance(other, Measurement):
            new_val = self._value / other._value
            rel_err = (self.rel**2 + other.rel**2) ** 0.5
            new_err = abs(new_val.magnitude) * rel_err
            return Measurement(new_val, new_err)
        if isinstance(other, int | float):
            return Measurement(self._value / other, self._error / abs(other))
        return NotImplemented


# --- Context stub ---


class Context:
    """Stub for pint Context compatibility.

    Contexts allow additional conversions (e.g., spectroscopy:
    wavelength to frequency). This is a minimal stub that accepts
    the API but does not perform context-based conversions.
    """

    def __init__(self, name: str = "", **kwargs: Any) -> None:
        self.name = name
        self._transforms: dict[tuple[str, str], Callable[..., Any]] = {}
        self._defaults = dict(kwargs)

    def add_transformation(
        self,
        src: str,
        dst: str,
        func: Callable[..., Any],
    ) -> None:
        """Register a transformation function."""
        self._transforms[(src, dst)] = func

    @staticmethod
    def from_lines(lines: list[str], to_base_func: Any = None) -> Context:  # noqa: ARG004
        """Create a Context from definition lines."""
        name = lines[0] if lines else ""
        return Context(name)


# --- Group stub ---


class Group:
    """Stub for pint Group compatibility."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._units: set[str] = set()

    def add_units(self, *units: str) -> None:
        self._units.update(units)

    @property
    def members(self) -> frozenset[str]:
        return frozenset(self._units)


# --- System stub ---


class System:
    """Stub for pint System compatibility."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._base_units: dict[str, str] = {}

    @property
    def base_units(self) -> dict[str, str]:
        return self._base_units


def _is_array_like(value: Any) -> bool:
    """Check if value is a numpy array or array-like (not a scalar/string)."""
    try:
        import numpy as np  # noqa: PLC0415

        return isinstance(value, np.ndarray | list) and not isinstance(value, str)
    except ImportError:
        return False


def make_quantity(
    ureg: UnitRegistry,
    value: Any,
    units: str | None = None,
) -> Any:
    """Create a Quantity, using ArrayQuantity for numpy arrays."""
    if _is_array_like(value) and ArrayQuantity is not None:
        units_str = units if units is not None else "dimensionless"
        return ArrayQuantity(value, units_str, ureg)
    return ureg.Quantity(value, units)


# --- Monkey-patch UnitRegistry with Python-level methods ---


def _ureg_wraps(
    self: UnitRegistry,
    ret: str | Unit | None,
    args: tuple[str | Unit | None, ...] | None = None,
    strict: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to wrap a function for automatic unit conversion."""
    return wraps(self, ret, args, strict)


def _ureg_check(
    self: UnitRegistry,
    *args: str | None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to check argument dimensions before calling."""
    return check(self, *args)


def _ureg_measurement(
    self: UnitRegistry,
    value: float,
    error: float,
    units: str | None = None,
) -> Measurement:
    """Create a Measurement."""
    if units is not None:
        q = self.Quantity(value, units)
    else:
        q = self.Quantity(value, "dimensionless")
    return Measurement(q, error)


def _ureg_setup_matplotlib(
    self: UnitRegistry,
    enable: bool = True,
) -> None:
    """Stub for matplotlib integration."""


def _ureg_context(
    self: UnitRegistry,  # noqa: ARG001
    *contexts: str | Context,  # noqa: ARG001
    **kwargs: Any,  # noqa: ARG001
) -> Any:
    """Stub for context manager."""
    return contextlib.nullcontext()


def _ureg_add_context(
    self: UnitRegistry,
    context: Context,
) -> None:
    """Stub for adding a context."""


def _ureg_remove_context(
    self: UnitRegistry,
    name: str,
) -> None:
    """Stub for removing a context."""


def _ureg_enable_contexts(
    self: UnitRegistry,
    *names: str,
    **kwargs: Any,
) -> None:
    """Stub for enabling contexts."""


def _ureg_disable_contexts(
    self: UnitRegistry,
    *names: str,
) -> None:
    """Stub for disabling contexts."""


def _ureg_with_context(
    self: UnitRegistry,  # noqa: ARG001
    *names: str,  # noqa: ARG001
    **kwargs: Any,  # noqa: ARG001
) -> Any:
    """Stub for context manager."""
    return contextlib.nullcontext()


def _ureg_get_group(self: UnitRegistry, name: str) -> Group:  # noqa: ARG001
    """Stub for getting a unit group."""
    return Group(name)


def _ureg_get_system(self: UnitRegistry, name: str) -> System:  # noqa: ARG001
    """Stub for getting a unit system."""
    return System(name)


def _ureg_parse_units_as_container(
    self: UnitRegistry,
    units: str,
) -> dict[str, float]:
    """Parse units into a container-like dict."""
    u = self.parse_units(units)
    return dict(u._units_dict())


def _ureg_parse_pattern(
    self: UnitRegistry,  # noqa: ARG001
    pattern: str,
    input_string: str,
) -> Any:
    """Stub for parse_pattern."""
    import re  # noqa: PLC0415

    return re.findall(pattern, input_string)


def _ureg_pi_theorem(
    self: UnitRegistry,  # noqa: ARG001
    quantities: dict[str, str],  # noqa: ARG001
) -> list[dict[str, float]]:
    """Stub for Buckingham Pi theorem."""
    return []


# Attach methods to UnitRegistry
UnitRegistry.wraps = _ureg_wraps  # type: ignore[attr-defined]
UnitRegistry.check = _ureg_check  # type: ignore[attr-defined]
UnitRegistry.Measurement = _ureg_measurement  # type: ignore[attr-defined]
UnitRegistry.setup_matplotlib = _ureg_setup_matplotlib  # type: ignore[attr-defined]
UnitRegistry.context = _ureg_context  # type: ignore[attr-defined]
UnitRegistry.add_context = _ureg_add_context  # type: ignore[attr-defined]
UnitRegistry.remove_context = _ureg_remove_context  # type: ignore[attr-defined]
UnitRegistry.enable_contexts = _ureg_enable_contexts  # type: ignore[attr-defined]
UnitRegistry.disable_contexts = _ureg_disable_contexts  # type: ignore[attr-defined]
UnitRegistry.with_context = _ureg_with_context  # type: ignore[attr-defined]
UnitRegistry.get_group = _ureg_get_group  # type: ignore[attr-defined]
UnitRegistry.get_system = _ureg_get_system  # type: ignore[attr-defined]
UnitRegistry.parse_units_as_container = _ureg_parse_units_as_container  # type: ignore[attr-defined]
UnitRegistry.parse_pattern = _ureg_parse_pattern  # type: ignore[attr-defined]
UnitRegistry.pi_theorem = _ureg_pi_theorem  # type: ignore[attr-defined]

# format_babel stub on Quantity and Unit
Quantity.format_babel = lambda self, locale="en": str(self)  # type: ignore[attr-defined]  # noqa: ARG005
Unit.format_babel = lambda self, locale="en": str(self)  # type: ignore[attr-defined]  # noqa: ARG005
