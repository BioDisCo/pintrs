"""pintrs - Fast Rust-based Python extension for physical unit manipulation."""

# pyright: reportPrivateUsage=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportAttributeAccessIssue=false, reportUnnecessaryIsInstance=false, reportUnnecessaryComparison=false

from __future__ import annotations

import contextlib
import functools
import sys
from typing import TYPE_CHECKING

from pintrs import _core as _core_module
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
from pintrs.context import ActiveContexts, Context
from pintrs.definitions import (
    build_groups_from_definitions,
    build_systems_from_definitions,
)
from pintrs.group import Group, _build_builtin_groups
from pintrs.logarithmic import LogarithmicQuantity
from pintrs.numpy_support import ArrayQuantity
from pintrs.pandas_support import PintArray, PintDtype
from pintrs.system import System

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

__all__ = [
    "ArrayQuantity",
    "Context",
    "DefinitionSyntaxError",
    "DimensionalityError",
    "Group",
    "LogarithmicQuantity",
    "Measurement",
    "OffsetUnitCalculusError",
    "PintArray",
    "PintDtype",
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
sys.modules.setdefault("_core", _core_module)

# --- Application registry singleton ---

_application_registry: UnitRegistry | None = None
_original_define = UnitRegistry.define


def set_application_registry(ureg: UnitRegistry) -> None:
    """Set the application-wide default registry."""
    global _application_registry  # noqa: PLW0603
    _application_registry = ureg


def get_application_registry() -> UnitRegistry:
    """Get the application-wide default registry, creating one if needed."""
    global _application_registry  # noqa: PLW0603
    if _application_registry is None:
        _application_registry = UnitRegistry()
        _ensure_registry_initialized(_application_registry)
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
                for val, unit_spec in zip(values, args):
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
            for val, expected_dim in zip(values, args):
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


# --- GenericQuantity (for Decimal, Fraction, and other non-float magnitudes) ---


class GenericQuantity:
    """A Quantity whose magnitude is a non-float type (Decimal, Fraction, etc.).

    Uses Rust-backed Unit for unit tracking, but keeps magnitude in Python
    to preserve the original type.
    """

    def __init__(
        self,
        magnitude: Any,
        units: str,
        registry: UnitRegistry | None = None,
    ) -> None:
        self._magnitude = magnitude
        self._units_str = units
        self._registry: UnitRegistry = registry or UnitRegistry()
        self._unit_obj: Unit = self._registry.Unit(units)

    @property
    def magnitude(self) -> Any:
        return self._magnitude

    @property
    def m(self) -> Any:
        return self._magnitude

    @property
    def units(self) -> Unit:
        return self._unit_obj

    @property
    def u(self) -> Unit:
        return self._unit_obj

    @property
    def dimensionality(self) -> str:
        return str(self._unit_obj.dimensionality)

    @property
    def dimensionless(self) -> bool:
        return bool(self._unit_obj.dimensionless)

    def to(self, units: str) -> GenericQuantity:
        factor = self._registry._get_conversion_factor(self._units_str, units)
        mag_type = type(self._magnitude)
        new_mag = self._magnitude * mag_type(str(factor))
        return GenericQuantity(new_mag, units, self._registry)

    def to_base_units(self) -> GenericQuantity:
        factor, unit_str = self._registry._get_root_units(self._units_str)
        mag_type = type(self._magnitude)
        new_mag = self._magnitude * mag_type(str(factor))
        return GenericQuantity(new_mag, unit_str, self._registry)

    def to_root_units(self) -> GenericQuantity:
        return self.to_base_units()

    def is_compatible_with(self, other: str | GenericQuantity | Quantity) -> bool:
        scalar = self._registry._scalar_quantity(1.0, self._units_str)
        if isinstance(other, str):
            return bool(scalar.is_compatible_with(other))
        if isinstance(other, GenericQuantity):
            return bool(scalar.is_compatible_with(other._units_str))
        return bool(scalar.is_compatible_with(other))

    def __add__(self, other: Any) -> GenericQuantity:
        if isinstance(other, GenericQuantity):
            if other._units_str != self._units_str:
                other = other.to(self._units_str)
            return GenericQuantity(
                self._magnitude + other._magnitude,
                self._units_str,
                self._registry,
            )
        return GenericQuantity(
            self._magnitude + other,
            self._units_str,
            self._registry,
        )

    def __radd__(self, other: Any) -> GenericQuantity:
        return self.__add__(other)

    def __sub__(self, other: Any) -> GenericQuantity:
        if isinstance(other, GenericQuantity):
            if other._units_str != self._units_str:
                other = other.to(self._units_str)
            return GenericQuantity(
                self._magnitude - other._magnitude,
                self._units_str,
                self._registry,
            )
        return GenericQuantity(
            self._magnitude - other,
            self._units_str,
            self._registry,
        )

    def __rsub__(self, other: Any) -> GenericQuantity:
        result = self.__sub__(other)
        return GenericQuantity(
            -result._magnitude,
            result._units_str,
            result._registry,
        )

    def __mul__(self, other: Any) -> GenericQuantity:
        if isinstance(other, GenericQuantity):
            new_units = str(self._unit_obj * other._unit_obj)
            return GenericQuantity(
                self._magnitude * other._magnitude,
                new_units,
                self._registry,
            )
        return GenericQuantity(
            self._magnitude * other,
            self._units_str,
            self._registry,
        )

    def __rmul__(self, other: Any) -> GenericQuantity:
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> GenericQuantity:
        if isinstance(other, GenericQuantity):
            new_units = str(self._unit_obj / other._unit_obj)
            return GenericQuantity(
                self._magnitude / other._magnitude,
                new_units,
                self._registry,
            )
        return GenericQuantity(
            self._magnitude / other,
            self._units_str,
            self._registry,
        )

    def __rtruediv__(self, other: Any) -> GenericQuantity:
        inv_units = str(
            self._registry.Unit("dimensionless") / self._unit_obj,
        )
        return GenericQuantity(
            other / self._magnitude,
            inv_units,
            self._registry,
        )

    def __pow__(self, exp: Any) -> GenericQuantity:
        new_units = str(self._unit_obj ** float(exp))
        return GenericQuantity(
            self._magnitude**exp,
            new_units,
            self._registry,
        )

    def __neg__(self) -> GenericQuantity:
        return GenericQuantity(-self._magnitude, self._units_str, self._registry)

    def __abs__(self) -> GenericQuantity:
        return GenericQuantity(abs(self._magnitude), self._units_str, self._registry)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, GenericQuantity):
            if other._units_str != self._units_str:
                other = other.to(self._units_str)
            return bool(self._magnitude == other._magnitude)
        return False

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, GenericQuantity):
            if other._units_str != self._units_str:
                other = other.to(self._units_str)
            return bool(self._magnitude < other._magnitude)
        return bool(self._magnitude < other)

    def __le__(self, other: Any) -> bool:
        return self == other or self < other

    def __gt__(self, other: Any) -> bool:
        if isinstance(other, GenericQuantity):
            if other._units_str != self._units_str:
                other = other.to(self._units_str)
            return bool(self._magnitude > other._magnitude)
        return bool(self._magnitude > other)

    def __ge__(self, other: Any) -> bool:
        return self == other or self > other

    def __float__(self) -> float:
        return float(self._magnitude)

    def __int__(self) -> int:
        return int(self._magnitude)

    def __bool__(self) -> bool:
        return bool(self._magnitude)

    def __repr__(self) -> str:
        return f"<Quantity({self._magnitude}, '{self._units_str}')>"

    def __str__(self) -> str:
        return f"{self._magnitude} {self._units_str}"

    def __format__(self, spec: str) -> str:
        if not spec:
            return str(self)
        # Delegate unit formatting to the Rust Unit
        q = self._registry._scalar_quantity(float(self._magnitude), self._units_str)
        return format(q, spec)

    def __hash__(self) -> int:
        return hash((self._magnitude, self._units_str))

    def __reduce__(self) -> tuple[Any, ...]:
        return (GenericQuantity, (self._magnitude, self._units_str))

    def check(self, dimension: str) -> bool:
        return bool(
            self._registry._scalar_quantity(1.0, self._units_str).check(dimension),
        )


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

    def __add__(self, other: Measurement | Quantity | float) -> Measurement:
        if isinstance(other, Measurement):
            new_val = self._value + other._value
            new_err = (self._error**2 + other._error**2) ** 0.5
            return Measurement(new_val, new_err)
        if isinstance(other, Quantity):
            return Measurement(self._value + other, self._error)
        if isinstance(other, (int, float)):
            return Measurement(self._value + other, self._error)
        return NotImplemented

    def __radd__(self, other: Quantity | float) -> Measurement:
        return self.__add__(other)

    def __sub__(self, other: Measurement | Quantity | float) -> Measurement:
        if isinstance(other, Measurement):
            new_val = self._value - other._value
            new_err = (self._error**2 + other._error**2) ** 0.5
            return Measurement(new_val, new_err)
        if isinstance(other, Quantity):
            return Measurement(self._value - other, self._error)
        if isinstance(other, (int, float)):
            return Measurement(self._value - other, self._error)
        return NotImplemented

    def __rsub__(self, other: Quantity | float) -> Measurement:
        if isinstance(other, Quantity):
            return Measurement(other - self._value, self._error)
        if isinstance(other, (int, float)):
            return Measurement(other - self._value, self._error)
        return NotImplemented

    def __mul__(self, other: Measurement | float) -> Measurement:
        if isinstance(other, Measurement):
            new_val = self._value * other._value
            rel_err = (self.rel**2 + other.rel**2) ** 0.5
            new_err = abs(new_val.magnitude) * rel_err
            return Measurement(new_val, new_err)
        if isinstance(other, (int, float)):
            return Measurement(self._value * other, self._error * abs(other))
        return NotImplemented

    def __rmul__(self, other: float) -> Measurement:
        return self.__mul__(other)

    def __truediv__(self, other: Measurement | float) -> Measurement:
        if isinstance(other, Measurement):
            new_val = self._value / other._value
            rel_err = (self.rel**2 + other.rel**2) ** 0.5
            new_err = abs(new_val.magnitude) * rel_err
            return Measurement(new_val, new_err)
        if isinstance(other, (int, float)):
            return Measurement(self._value / other, self._error / abs(other))
        return NotImplemented

    def __rtruediv__(self, other: float) -> Measurement:
        if isinstance(other, (int, float)):
            new_val = other / self._value
            new_err = abs(new_val.magnitude) * self.rel
            return Measurement(new_val, new_err)
        return NotImplemented


# --- Delta temperature units ---
# These are temperature differences, not absolute temperatures.
# In pint, adding two absolute temperatures is an error;
# only delta + absolute or delta + delta is valid.

_DELTA_UNIT_DEFS = [
    "delta_degree_Celsius = kelvin; offset: 0 = delta_degC = delta_°C",
    "delta_degree_Fahrenheit = 5 / 9 * kelvin; offset: 0 = delta_degF = delta_°F",
    "delta_degree_Rankine = 5 / 9 * kelvin; offset: 0 = delta_degR",
    "delta_degree_Reaumur = 5 / 4 * kelvin; offset: 0 = delta_degRe",
]

_OFFSET_UNITS = frozenset(
    {
        "degree_Celsius",
        "degree_Fahrenheit",
        "degree_Rankine",
        "degree_Reaumur",
    }
)


def _register_delta_units(ureg: UnitRegistry) -> None:
    """Register delta temperature units on a registry."""
    for defn in _DELTA_UNIT_DEFS:
        with contextlib.suppress(RedefinitionError, DefinitionSyntaxError):
            ureg.define(defn)


# --- Group & System ---
# Imported from pintrs.group and pintrs.system


def _is_array_like(value: Any) -> bool:
    """Check if value is a numpy array or array-like (not a scalar/string)."""
    try:
        import numpy as np  # noqa: PLC0415

        return isinstance(value, (np.ndarray, list)) and not isinstance(value, str)
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


class _Dimensionality(dict):  # type: ignore[type-arg]
    """Dict-like dimensionality object for pint compatibility.

    Pint returns UnitsContainer({'[length]': 1, '[time]': -1}) from
    .dimensionality. This class parses pintrs's string representation
    into the same dict-like interface.
    """

    def __init__(self, dim_string: str) -> None:
        super().__init__()
        self._string = dim_string
        if dim_string in ("dimensionless", "[]", "") or not dim_string:
            return
        normalized = dim_string.replace(" ", "").replace("**", "^")
        if "/" in normalized:
            num, den = normalized.split("/", 1)
        else:
            num, den = normalized, ""
        for part in num.split("*"):
            if not part or part == "1":
                continue
            if "^" in part:
                name, exp = part.split("^", 1)
                self[name] = float(exp)
            else:
                self[part] = 1.0
        for part in den.split("*"):
            if not part or part == "1":
                continue
            if "^" in part:
                name, exp = part.split("^", 1)
                self[name] = -float(exp)
            else:
                self[part] = -1.0

    def __str__(self) -> str:
        return self._string

    def __repr__(self) -> str:
        return self._string

    def __hash__(self) -> int:  # type: ignore[override]
        return hash(frozenset(self.items()))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self._string == other
        return super().__eq__(other)

    def __contains__(self, item: object) -> bool:
        if isinstance(item, str) and item not in dict.keys(self):
            return item in self._string
        return super().__contains__(item)

    def __bool__(self) -> bool:
        return len(self) > 0


def _parse_dimensionality(dim_string: str) -> _Dimensionality:
    """Parse a dimensionality string into a dict-like object."""
    return _Dimensionality(dim_string)


_original_q_dimensionality = Quantity.dimensionality  # type: ignore[attr-defined]


def _q_dimensionality_compat(self: Any) -> _Dimensionality:
    dim_str: str = _original_q_dimensionality.__get__(self)  # type: ignore[misc]
    return _parse_dimensionality(dim_str)


Quantity.dimensionality = property(_q_dimensionality_compat)  # type: ignore[attr-defined,assignment]

_original_u_dimensionality = Unit.dimensionality  # type: ignore[attr-defined]


def _u_dimensionality_compat(self: Any) -> _Dimensionality:
    dim_str: str = _original_u_dimensionality.__get__(self)  # type: ignore[misc]
    return _parse_dimensionality(dim_str)


Unit.dimensionality = property(_u_dimensionality_compat)  # type: ignore[attr-defined,assignment]


def _q_REGISTRY(self: Any) -> UnitRegistry:  # noqa: N802
    return self._registry  # type: ignore[no-any-return]


Quantity._REGISTRY = property(_q_REGISTRY)  # type: ignore[attr-defined]


_original_get_dimensionality = UnitRegistry.get_dimensionality


def _ureg_get_dimensionality_compat(self: Any, unit: str) -> _Dimensionality:
    dim_str: str = _original_get_dimensionality(self, unit)
    return _parse_dimensionality(dim_str)


UnitRegistry.get_dimensionality = _ureg_get_dimensionality_compat  # type: ignore[attr-defined,assignment]


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
    self: UnitRegistry,  # noqa: ARG001
    enable: bool = True,
) -> None:
    """Enable or disable matplotlib integration for pintrs Quantities."""
    from pintrs.matplotlib_support import setup_matplotlib  # noqa: PLC0415

    setup_matplotlib(enable)


_registry_contexts: dict[int, ActiveContexts] = {}
_initialized_registries: set[int] = set()


def _ensure_registry_initialized(ureg: UnitRegistry) -> None:
    """Lazily register delta units and other extras on a registry."""
    key = id(ureg)
    if key not in _initialized_registries:
        _initialized_registries.add(key)
        _register_delta_units(ureg)


def _get_active_contexts(ureg: UnitRegistry) -> ActiveContexts:
    """Get or create the ActiveContexts for a registry."""
    key = id(ureg)
    ctx = _registry_contexts.get(key)
    if ctx is None:
        ctx = ActiveContexts()
        _registry_contexts[key] = ctx
    return ctx


@contextlib.contextmanager
def _ureg_context(
    self: UnitRegistry,
    *contexts: str | Context,
    **kwargs: Any,  # noqa: ARG001
) -> Any:
    """Context manager that temporarily enables contexts."""
    _ensure_registry_initialized(self)
    active = _get_active_contexts(self)
    prev_state = getattr(_context_state, "current", None)
    with active(*contexts):
        _context_state.current = (self, active)
        try:
            yield
        finally:
            _context_state.current = prev_state


def _ureg_add_context(
    self: UnitRegistry,  # noqa: ARG001
    context: Context,
) -> None:
    """Register a context with the registry."""
    if isinstance(context, Context):
        Context._REGISTRY[context.name] = context


def _ureg_remove_context(
    self: UnitRegistry,  # noqa: ARG001
    name: str,
) -> None:
    """Remove a context from the registry."""
    Context._REGISTRY.pop(name, None)


def _ureg_enable_contexts(
    self: UnitRegistry,
    *names: str,
    **kwargs: Any,  # noqa: ARG001
) -> None:
    """Enable contexts on the registry."""
    active = _get_active_contexts(self)
    active.enable(*names)
    _context_state.current = (self, active)


def _ureg_disable_contexts(
    self: UnitRegistry,
    *names: str,
) -> None:
    """Disable contexts on the registry."""
    active = _get_active_contexts(self)
    active.disable(*names)
    if active.active:
        _context_state.current = (self, active)
    elif getattr(_context_state, "current", None) == (self, active):
        _context_state.current = None


def _ureg_with_context(
    self: UnitRegistry,
    *names: str,
    **kwargs: Any,  # noqa: ARG001
) -> Any:
    """Context manager that temporarily enables contexts."""
    return _ureg_context(self, *names)


def _ureg_get_group(self: UnitRegistry, name: str) -> Group:
    """Get a unit group by name, building built-in groups if needed."""
    _ensure_registry_initialized(self)
    build_groups_from_definitions(self)
    _build_builtin_groups(self)
    grp = Group.get(name)
    if grp is None:
        grp = Group(name)
    return grp


def _ureg_get_system(self: UnitRegistry, name: str) -> System:
    """Get a unit system by name."""
    _ensure_registry_initialized(self)
    build_systems_from_definitions(self)
    sys = System.get(name)
    if sys is None:
        sys = System(name)
    return sys


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


_EPSILON = 1e-10


def _ureg_pi_theorem(
    self: UnitRegistry,
    quantities: dict[str, str],
) -> list[dict[str, float]]:
    """Apply the Buckingham Pi theorem to find dimensionless groups.

    Args:
        quantities: Mapping of variable names to unit strings.

    Returns:
        List of dicts mapping variable names to exponents forming
        dimensionless combinations.
    """
    if not quantities:
        return []

    names = list(quantities.keys())
    all_dims: set[str] = set()
    dim_maps: list[dict[str, float]] = []

    for unit_str in quantities.values():
        _factor, base_unit = self.get_base_units(unit_str)
        base_d = dict(base_unit._units_dict())
        dim_maps.append(base_d)
        all_dims.update(base_d.keys())

    if not all_dims:
        return [{name: 1.0} for name in names]

    dim_list = sorted(all_dims)
    n_vars = len(names)
    n_dims = len(dim_list)

    # Build the dimension matrix: rows = dimensions, cols = variables
    matrix = []
    for dim in dim_list:
        row = [dim_maps[j].get(dim, 0.0) for j in range(n_vars)]
        matrix.append(row)

    # Find null space using Gaussian elimination
    null_vectors = _null_space(matrix, n_dims, n_vars)

    result = []
    for vec in null_vectors:
        normalized_vec = vec
        for entry in normalized_vec:
            if abs(entry) > _EPSILON:
                if entry < 0:
                    normalized_vec = [-x for x in normalized_vec]
                break
        group = {}
        for i, name in enumerate(names):
            if abs(normalized_vec[i]) > _EPSILON:
                group[name] = normalized_vec[i]
        if group:
            result.append(group)

    return result


def _null_space(
    matrix: list[list[float]],
    n_rows: int,
    n_cols: int,
) -> list[list[float]]:
    """Compute null space of a matrix via Gaussian elimination.

    Args:
        matrix: n_rows x n_cols matrix.
        n_rows: Number of rows.
        n_cols: Number of columns.

    Returns:
        List of null space vectors (each of length n_cols).
    """
    # Augmented matrix for row reduction
    aug = [row[:] for row in matrix]

    pivot_cols: list[int] = []
    row_idx = 0

    for col in range(n_cols):
        # Find pivot
        pivot = None
        for r in range(row_idx, n_rows):
            if abs(aug[r][col]) > _EPSILON:
                pivot = r
                break
        if pivot is None:
            continue

        # Swap rows
        aug[row_idx], aug[pivot] = aug[pivot], aug[row_idx]

        # Scale pivot row
        scale = aug[row_idx][col]
        aug[row_idx] = [x / scale for x in aug[row_idx]]

        # Eliminate column
        for r in range(n_rows):
            if r != row_idx and abs(aug[r][col]) > _EPSILON:
                factor = aug[r][col]
                aug[r] = [aug[r][c] - factor * aug[row_idx][c] for c in range(n_cols)]

        pivot_cols.append(col)
        row_idx += 1

    # Free variables are columns not in pivot_cols
    free_cols = [c for c in range(n_cols) if c not in pivot_cols]
    rank = len(pivot_cols)

    vectors = []
    for free_col in free_cols:
        vec = [0.0] * n_cols
        vec[free_col] = 1.0
        for i, pc in enumerate(pivot_cols):
            if i < rank:
                vec[pc] = -aug[i][free_col]
        vectors.append(vec)

    return vectors


# --- Context-aware conversion on Quantity ---

_original_quantity_to = Quantity.to

# Thread-local storage for the active registry+contexts during context blocks
import threading  # noqa: E402

_context_state = threading.local()


def _get_context_registry() -> tuple[UnitRegistry, ActiveContexts] | None:
    """Get the currently active registry and contexts, if any."""
    return getattr(_context_state, "current", None)


def _quantity_to_with_context(self: Any, units: str) -> Any:
    """Quantity.to() that falls back to active context transformations."""
    try:
        return _original_quantity_to(self, units)
    except DimensionalityError:
        state = _get_context_registry()
        if state is None:
            raise
        ureg, active = state
        if not active.active:
            raise
        src_dim = self.dimensionality
        dst_dim = ureg.get_dimensionality(units)
        result = active.find_transform(ureg, self, src_dim, dst_dim)
        if result is None:
            raise
        if hasattr(result, "to"):
            return result.to(units)
        base_q = ureg.Quantity(float(result), _base_unit_for_dim(str(dst_dim)))
        return _original_quantity_to(base_q, units)


def _base_unit_for_dim(dim: str) -> str:
    """Map a dimensionality string to its SI base unit."""
    mapping: dict[str, str] = {
        "[length]": "meter",
        "[time]": "second",
        "[mass]": "kilogram",
        "[temperature]": "kelvin",
        "[current]": "ampere",
        "[substance]": "mole",
        "[luminosity]": "candela",
        "1 / [time]": "hertz",
        "[length] ** 2 * [mass] / [time] ** 2": "joule",
        "[mass] / [current] ** 2 / [time] ** 3": "watt",
        "[mass] / [length] / [time] ** 2": "pascal",
    }
    return mapping.get(dim, "dimensionless")


Quantity.to = _quantity_to_with_context  # type: ignore[attr-defined,assignment]

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

# --- Config flags (per-registry, stored externally) ---

_registry_configs: dict[int, dict[str, Any]] = {}

_DEFAULT_CONFIG: dict[str, Any] = {
    "autoconvert_offset_to_baseunit": False,
    "autoconvert_to_preferred": False,
    "cache_folder": None,
    "case_sensitive": True,
    "default_as_delta": True,
    "default_system": "mks",
    "fmt_locale": None,
    "force_ndarray": False,
    "force_ndarray_like": False,
    "non_int_type": float,
    "separate_format_defaults": None,
    "preprocessors": [],
    "mpl_formatter": "{:P}",
}

_CONFIG_KEYS = frozenset(_DEFAULT_CONFIG)


def _get_config(ureg: UnitRegistry) -> dict[str, Any]:
    key = id(ureg)
    cfg = _registry_configs.get(key)
    if cfg is None:
        cfg = dict(_DEFAULT_CONFIG)
        cfg["preprocessors"] = []
        init_kwargs = getattr(ureg, "_init_kwargs", None)
        if init_kwargs is not None:
            for k, v in init_kwargs.items():
                if k in _CONFIG_KEYS:
                    cfg[k] = v
        _registry_configs[key] = cfg
        _update_special_flag(ureg, cfg)
    return cfg


def _update_special_flag(ureg: UnitRegistry, cfg: dict[str, Any]) -> None:
    """Update fast-path bypass flag when special Quantity config is active."""
    has_special = (
        bool(cfg.get("force_ndarray")) or cfg.get("non_int_type", float) is not float
    )
    key = id(ureg)
    if has_special:
        _special_qty_registries.add(key)
    else:
        _special_qty_registries.discard(key)


def _make_config_property(name: str) -> property:
    def getter(self: UnitRegistry) -> Any:
        return _get_config(self)[name]

    def setter(self: UnitRegistry, value: Any) -> None:
        cfg = _get_config(self)
        cfg[name] = value
        if name in ("force_ndarray", "non_int_type"):
            _update_special_flag(self, cfg)

    return property(getter, setter)


for _cfg_name in _CONFIG_KEYS:
    setattr(UnitRegistry, _cfg_name, _make_config_property(_cfg_name))


def _ureg_set_fmt_locale(self: UnitRegistry, locale: str | None) -> None:
    """Set the locale for formatting."""
    _get_config(self)["fmt_locale"] = locale


UnitRegistry.set_fmt_locale = _ureg_set_fmt_locale  # type: ignore[attr-defined]


# --- Smart Quantity constructor (dispatches to ArrayQuantity for arrays) ---


def _is_duck_array(value: Any) -> bool:
    """Check if value is an array-like that should become ArrayQuantity."""
    try:
        import numpy as np  # noqa: PLC0415

        if isinstance(value, np.ndarray):
            return True
    except ImportError:
        pass
    if isinstance(value, (list, tuple)):
        return True
    return hasattr(value, "__array_function__") or hasattr(value, "__array_ufunc__")


_special_qty_registries: set[int] = set()


def _ureg_quantity(self: UnitRegistry, value: Any, units: Any = None) -> Any:  # noqa: PLR0911
    """Create a Quantity, dispatching to ArrayQuantity for array-like inputs."""
    # Fast path: float/int scalar (most common case) -> straight to Rust
    vtype = type(value)
    if (vtype is float or vtype is int) and id(self) not in _special_qty_registries:
        return self._f64_quantity(
            float(value),
            str(units) if units is not None else None,
        )

    # Quantity(Quantity, new_units) -> convert (delegate to Rust)
    if vtype is Quantity:
        return self._scalar_quantity(value, str(units) if units is not None else None)

    # String parsing -> delegate to Rust
    if vtype is str:
        return self._scalar_quantity(value, None)

    from pintrs.numpy_support import ArrayQuantity as _ArrayQuantity  # noqa: PLC0415

    # Quantity(ArrayQuantity, new_units) -> convert
    if isinstance(value, _ArrayQuantity):
        if units is not None:
            return value.to(str(units))
        return value

    # Quantity(Unit) -> Quantity(1, unit)
    if isinstance(value, Unit):
        return self._scalar_quantity(1.0, str(value))

    # Array-like -> ArrayQuantity
    if _is_duck_array(value):
        try:
            import numpy as np  # noqa: PLC0415

            units_str = str(units) if units is not None else "dimensionless"
            arr = np.asarray(value) if not isinstance(value, np.ndarray) else value
            return _ArrayQuantity(arr, units_str, self)
        except (ImportError, TypeError, ValueError):
            pass

    # force_ndarray: wrap scalar in 0-d array
    cfg = _get_config(self)
    if cfg.get("force_ndarray"):
        try:
            import numpy as np  # noqa: PLC0415

            units_str = str(units) if units is not None else "dimensionless"
            return _ArrayQuantity(np.asarray(value), units_str, self)
        except (ImportError, TypeError, ValueError):
            pass

    # non_int_type: preserve Decimal/Fraction magnitudes
    non_int = cfg.get("non_int_type", float)
    if non_int is not float and isinstance(value, non_int):
        units_str = str(units) if units is not None else "dimensionless"
        return GenericQuantity(value, units_str, self)

    # Scalar -> Rust Quantity
    return self._scalar_quantity(value, str(units) if units is not None else None)


UnitRegistry.Quantity = _ureg_quantity  # type: ignore[attr-defined]

# --- Class references on UnitRegistry ---

UnitRegistry.Context = Context  # type: ignore[attr-defined]
UnitRegistry.Group = Group  # type: ignore[attr-defined]
UnitRegistry.System = System  # type: ignore[attr-defined]
UnitRegistry.UnitsContainer = _ureg_parse_units_as_container  # type: ignore[attr-defined]


# --- System lister (ureg.sys) ---


class _SystemLister:
    """Attribute-access lister for unit systems, matching pint's ureg.sys."""

    def __init__(self, ureg: UnitRegistry) -> None:
        self._ureg = ureg

    def __dir__(self) -> list[str]:
        return list(System._REGISTRY.keys())

    def __getattr__(self, name: str) -> System:
        return _ureg_get_system(self._ureg, name)

    def __repr__(self) -> str:
        return f"<SystemLister({list(System._REGISTRY.keys())})>"


def _ureg_sys_getter(self: UnitRegistry) -> _SystemLister:
    _ensure_registry_initialized(self)
    build_systems_from_definitions(self)
    return _SystemLister(self)


UnitRegistry.sys = property(_ureg_sys_getter)  # type: ignore[attr-defined,assignment]

# --- Formatter stub ---


class _Formatter:
    """Minimal formatter stub matching pint's Formatter interface."""

    def __init__(self) -> None:
        self.default_format: str = ""
        self.fmt_locale: str | None = None


def _ureg_formatter_getter(self: UnitRegistry) -> _Formatter:
    cfg = _get_config(self)
    f = _Formatter()
    f.fmt_locale = cfg["fmt_locale"]
    return f


UnitRegistry.formatter = property(_ureg_formatter_getter)  # type: ignore[attr-defined,assignment]


# --- Quantity: numpy-like properties and methods ---


def _q_ndim(self: Any) -> int:
    """Number of dimensions of the magnitude."""
    m = self.magnitude
    if hasattr(m, "ndim"):
        return m.ndim  # type: ignore[no-any-return]
    return 0


def _q_shape(self: Any) -> tuple[int, ...]:
    """Shape of the magnitude."""
    m = self.magnitude
    if hasattr(m, "shape"):
        return m.shape  # type: ignore[no-any-return]
    return ()


def _q_dtype(self: Any) -> Any:
    """Dtype of the magnitude."""
    m = self.magnitude
    if hasattr(m, "dtype"):
        return m.dtype
    try:
        import numpy as np  # noqa: PLC0415

        return np.dtype(type(m))
    except ImportError:
        return type(m)


def _q_real(self: Any) -> Any:
    """Real part of the quantity."""
    m = self.magnitude
    real_m = m.real if hasattr(m, "real") else m
    return self.__class__(real_m, str(self.units))


def _q_imag(self: Any) -> Any:
    """Imaginary part of the quantity."""
    m = self.magnitude
    imag_m = m.imag if hasattr(m, "imag") else 0
    return self.__class__(imag_m, str(self.units))


def _q_transpose(self: Any) -> Any:
    """Transpose of the magnitude."""
    m = self.magnitude
    if hasattr(m, "T"):
        return self.__class__(m.T, str(self.units))
    return self


Quantity.ndim = property(_q_ndim)  # type: ignore[attr-defined,assignment]
Quantity.shape = property(_q_shape)  # type: ignore[attr-defined,assignment]
Quantity.dtype = property(_q_dtype)  # type: ignore[attr-defined,assignment]
Quantity.real = property(_q_real)  # type: ignore[attr-defined,assignment]
Quantity.imag = property(_q_imag)  # type: ignore[attr-defined,assignment]
Quantity.T = property(_q_transpose)  # type: ignore[attr-defined,assignment]
Quantity.force_ndarray = False  # type: ignore[attr-defined]
Quantity.force_ndarray_like = False  # type: ignore[attr-defined]
Quantity.__array_priority__ = 21  # type: ignore[attr-defined]  # higher than ndarray (0)


def _q_clip(self: Any, min: Any = None, max: Any = None) -> Any:  # noqa: A002
    """Clip magnitude values."""
    m = self.magnitude
    if hasattr(m, "clip"):
        return self.__class__(m.clip(min, max), str(self.units))
    clipped = m
    if min is not None and clipped < min:
        clipped = min
    if max is not None and clipped > max:
        clipped = max
    return self.__class__(clipped, str(self.units))


def _q_dot(self: Any, other: Any) -> Any:
    """Dot product."""
    if hasattr(self.magnitude, "dot"):
        if hasattr(other, "magnitude"):
            return self.__class__(
                self.magnitude.dot(other.magnitude),
                str(self.units * other.units),
            )
        return self.__class__(self.magnitude.dot(other), str(self.units))
    msg = "Scalar quantities do not support dot product"
    raise AttributeError(msg)


def _q_prod(self: Any, axis: Any = None) -> Any:
    """Product of magnitude elements."""
    m = self.magnitude
    if hasattr(m, "prod"):
        return self.__class__(m.prod(axis=axis), str(self.units))
    return self


def _q_fill(self: Any, value: Any) -> None:
    """Fill magnitude with a scalar value."""
    m = self.magnitude
    if hasattr(m, "fill"):
        m.fill(value)
    else:
        msg = "Scalar quantities do not support fill"
        raise AttributeError(msg)


def _q_flat(self: Any) -> Any:
    """Flat iterator over magnitude values."""
    m = self.magnitude
    if hasattr(m, "flat"):
        for val in m.flat:
            yield self.__class__(val, str(self.units))
    else:
        yield self


def _q_put(self: Any, indices: Any, values: Any) -> None:
    """Set magnitude values at indices."""
    m = self.magnitude
    if hasattr(m, "put"):
        if hasattr(values, "magnitude"):
            values = values.magnitude
        m.put(indices, values)
    else:
        msg = "Scalar quantities do not support put"
        raise AttributeError(msg)


def _q_searchsorted(self: Any, v: Any, side: str = "left") -> Any:
    """Find indices for inserting values."""
    m = self.magnitude
    if hasattr(m, "searchsorted"):
        if hasattr(v, "magnitude"):
            v = v.to(str(self.units)).magnitude
        return m.searchsorted(v, side=side)
    msg = "Scalar quantities do not support searchsorted"
    raise AttributeError(msg)


def _q_tolist(self: Any) -> Any:
    """Convert magnitude to a Python list."""
    m = self.magnitude
    if hasattr(m, "tolist"):
        return m.tolist()
    return m


def _q_from_list(_cls: Any, lst: list[Any], units: str | None = None) -> Any:
    """Create a Quantity from a list of Quantities."""
    try:
        import numpy as np  # noqa: PLC0415
    except ImportError as e:
        msg = "from_list requires numpy"
        raise ImportError(msg) from e
    if not lst:
        return ArrayQuantity(np.array([]), units or "dimensionless")
    if units is None and hasattr(lst[0], "units"):
        units = str(lst[0].units)
    magnitudes = [q.to(units).magnitude if hasattr(q, "magnitude") else q for q in lst]
    return ArrayQuantity(np.array(magnitudes), units or "dimensionless")


def _q_from_sequence(cls: Any, seq: Any, units: str | None = None) -> Any:
    """Create a Quantity from a sequence of Quantities."""
    return _q_from_list(cls, list(seq), units)


Quantity.clip = _q_clip  # type: ignore[attr-defined]
Quantity.dot = _q_dot  # type: ignore[attr-defined]
Quantity.prod = _q_prod  # type: ignore[attr-defined]
Quantity.fill = _q_fill  # type: ignore[attr-defined]
Quantity.flat = property(_q_flat)  # type: ignore[attr-defined,assignment]
Quantity.put = _q_put  # type: ignore[attr-defined]
Quantity.searchsorted = _q_searchsorted  # type: ignore[attr-defined]
Quantity.tolist = _q_tolist  # type: ignore[attr-defined]
Quantity.from_list = classmethod(_q_from_list)  # type: ignore[attr-defined,arg-type]
Quantity.from_sequence = classmethod(_q_from_sequence)  # type: ignore[attr-defined,arg-type]


# --- Quantity: dask stubs ---


def _q_compute(self: Any, **kwargs: Any) -> Any:  # noqa: ARG001
    """Compute dask arrays. Returns self for non-dask magnitudes."""
    m = self.magnitude
    if hasattr(m, "compute"):
        return self.__class__(m.compute(), str(self.units))
    return self


def _q_persist(self: Any, **kwargs: Any) -> Any:  # noqa: ARG001
    """Persist dask arrays. Returns self for non-dask magnitudes."""
    m = self.magnitude
    if hasattr(m, "persist"):
        return self.__class__(m.persist(), str(self.units))
    return self


def _q_visualize(self: Any, **kwargs: Any) -> Any:
    """Visualize dask task graph."""
    m = self.magnitude
    if hasattr(m, "visualize"):
        return m.visualize(**kwargs)
    msg = "Magnitude does not support visualization (not a dask array)"
    raise AttributeError(msg)


Quantity.UnitsContainer = _ureg_parse_units_as_container  # type: ignore[attr-defined]
Quantity.compute = _q_compute  # type: ignore[attr-defined]
Quantity.persist = _q_persist  # type: ignore[attr-defined]
Quantity.visualize = _q_visualize  # type: ignore[attr-defined]


def _ureg_define(self: UnitRegistry, definition: str) -> None:
    """Validate obvious invalid/redefinition cases before delegating to Rust."""
    stripped = definition.strip()
    if not stripped or stripped.startswith("="):
        msg = f"Invalid definition syntax: {definition!r}"
        raise DefinitionSyntaxError(msg)
    name, sep, _rest = stripped.partition("=")
    if not sep:
        msg = f"Invalid definition syntax: {definition!r}"
        raise DefinitionSyntaxError(msg)
    unit_name = name.strip()
    if not unit_name:
        msg = f"Invalid definition syntax: {definition!r}"
        raise DefinitionSyntaxError(msg)
    if unit_name in self:
        msg = f"Cannot redefine {unit_name!r}"
        raise RedefinitionError(msg)
    _original_define(self, definition)


UnitRegistry.define = _ureg_define  # type: ignore[attr-defined,assignment]

# --- Babel/locale formatting ---


def _quantity_format_babel(self: Any, locale: str = "en") -> str:
    """Format a Quantity using Babel for locale-aware output.

    Args:
        locale: Babel locale string (e.g. "en_US", "fr_FR", "de_DE").

    Returns:
        Locale-formatted string like "1.000,5 Meter" (de_DE).
    """
    try:
        from babel.numbers import format_decimal  # noqa: PLC0415
    except ImportError:
        return str(self)
    mag_str = format_decimal(self.magnitude, locale=locale)
    return f"{mag_str} {self.units}"


def _unit_format_babel(self: Any, locale: str = "en") -> str:  # noqa: ARG001
    """Format a Unit using Babel for locale-aware output.

    Args:
        locale: Babel locale string.

    Returns:
        Locale-formatted unit string.
    """
    return str(self)


Quantity.format_babel = _quantity_format_babel  # type: ignore[attr-defined]
Unit.format_babel = _unit_format_babel  # type: ignore[attr-defined]

# --- Formatting modes ---

from pintrs.formatting import format_quantity as _format_quantity  # noqa: E402
from pintrs.formatting import format_unit as _format_unit  # noqa: E402


def _quantity_format(self: Any, spec: str) -> str:
    if not spec:
        return str(self)
    return _format_quantity(self, spec)


def _unit_format(self: Any, spec: str) -> str:
    if not spec:
        return str(self)
    return _format_unit(self, spec)


Quantity.__format__ = _quantity_format  # type: ignore[attr-defined,assignment]
Unit.__format__ = _unit_format  # type: ignore[attr-defined,assignment]


def _unit_systems(self: Unit) -> frozenset[str]:
    """Return the set of system names this unit belongs to."""
    from pintrs.system import System  # noqa: PLC0415

    # Use canonical names from the units dict
    unit_names = {name for name, _ in self._units_dict()}
    result: set[str] = set()
    for sys_name, system in System._REGISTRY.items():
        if unit_names & set(system.rules.values()):
            result.add(sys_name)
    return frozenset(result)


Unit.systems = property(_unit_systems)  # type: ignore[attr-defined,assignment]

# --- Wrap Quantity arithmetic to handle array operands ---
# Fast scalar types that should go straight to Rust
_SCALAR_TYPES = (int, float, Quantity)

_original_q_mul = Quantity.__mul__
_original_q_rmul = Quantity.__rmul__
_original_q_add = Quantity.__add__
_original_q_radd = Quantity.__radd__
_original_q_sub = Quantity.__sub__
_original_q_rsub = Quantity.__rsub__
_original_q_truediv = Quantity.__truediv__
_original_q_rtruediv = Quantity.__rtruediv__

# Cache numpy at module level
try:
    import numpy as _np_cached
except ImportError:
    _np_cached = None  # type: ignore[assignment]


def _to_arr(other: Any) -> Any:
    """Convert array-like to ndarray. Returns None if not array-like."""
    if _np_cached is not None and isinstance(other, _np_cached.ndarray):
        return other
    if isinstance(other, (list, tuple)):
        return _np_cached.asarray(other) if _np_cached is not None else None
    if hasattr(other, "__array_function__"):
        return _np_cached.asarray(other) if _np_cached is not None else None
    return None


def _q_array_mul(self: Any, other: Any) -> Any:
    if type(other) in _SCALAR_TYPES:
        return _original_q_mul(self, other)
    arr = _to_arr(other)
    if arr is not None:
        return ArrayQuantity(arr * self.magnitude, str(self.units), self._registry)
    return _original_q_mul(self, other)


def _q_array_rmul(self: Any, other: Any) -> Any:
    if type(other) in _SCALAR_TYPES:
        return _original_q_rmul(self, other)
    arr = _to_arr(other)
    if arr is not None:
        return ArrayQuantity(arr * self.magnitude, str(self.units), self._registry)
    return _original_q_rmul(self, other)


def _q_array_add(self: Any, other: Any) -> Any:
    if type(other) in _SCALAR_TYPES:
        return _original_q_add(self, other)
    arr = _to_arr(other)
    if arr is not None:
        return ArrayQuantity(arr + self.magnitude, str(self.units), self._registry)
    return _original_q_add(self, other)


def _q_array_radd(self: Any, other: Any) -> Any:
    if type(other) in _SCALAR_TYPES:
        return _original_q_radd(self, other)
    arr = _to_arr(other)
    if arr is not None:
        return ArrayQuantity(arr + self.magnitude, str(self.units), self._registry)
    return _original_q_radd(self, other)


def _q_array_sub(self: Any, other: Any) -> Any:
    if type(other) in _SCALAR_TYPES:
        return _original_q_sub(self, other)
    arr = _to_arr(other)
    if arr is not None:
        return ArrayQuantity(
            self.magnitude - arr,
            str(self.units),
            self._registry,
        )
    return _original_q_sub(self, other)


def _q_array_rsub(self: Any, other: Any) -> Any:
    if type(other) in _SCALAR_TYPES:
        return _original_q_rsub(self, other)
    arr = _to_arr(other)
    if arr is not None:
        return ArrayQuantity(
            arr - self.magnitude,
            str(self.units),
            self._registry,
        )
    return _original_q_rsub(self, other)


def _q_array_truediv(self: Any, other: Any) -> Any:
    if type(other) in _SCALAR_TYPES:
        return _original_q_truediv(self, other)
    arr = _to_arr(other)
    if arr is not None:
        return ArrayQuantity(
            self.magnitude / arr,
            str(self.units),
            self._registry,
        )
    return _original_q_truediv(self, other)


def _q_array_rtruediv(self: Any, other: Any) -> Any:
    if type(other) in _SCALAR_TYPES:
        return _original_q_rtruediv(self, other)
    arr = _to_arr(other)
    if arr is not None:
        inv_units = str(self._registry.Unit("dimensionless") / self.units)
        return ArrayQuantity(arr / self.magnitude, inv_units, self._registry)
    return _original_q_rtruediv(self, other)


Quantity.__mul__ = _q_array_mul  # type: ignore[attr-defined,assignment]
Quantity.__rmul__ = _q_array_rmul  # type: ignore[attr-defined,assignment]
Quantity.__add__ = _q_array_add  # type: ignore[attr-defined,assignment]
Quantity.__radd__ = _q_array_radd  # type: ignore[attr-defined,assignment]
Quantity.__sub__ = _q_array_sub  # type: ignore[attr-defined,assignment]
Quantity.__rsub__ = _q_array_rsub  # type: ignore[attr-defined,assignment]
Quantity.__truediv__ = _q_array_truediv  # type: ignore[attr-defined,assignment]
Quantity.__rtruediv__ = _q_array_rtruediv  # type: ignore[attr-defined,assignment]
