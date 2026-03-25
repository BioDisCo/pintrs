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
        base_q = ureg.Quantity(float(result), _base_unit_for_dim(dst_dim))
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
        _registry_configs[key] = cfg
    return cfg


def _make_config_property(name: str) -> property:
    def getter(self: UnitRegistry) -> Any:
        return _get_config(self)[name]

    def setter(self: UnitRegistry, value: Any) -> None:
        _get_config(self)[name] = value

    return property(getter, setter)


for _cfg_name in _CONFIG_KEYS:
    setattr(UnitRegistry, _cfg_name, _make_config_property(_cfg_name))


def _ureg_set_fmt_locale(self: UnitRegistry, locale: str | None) -> None:
    """Set the locale for formatting."""
    _get_config(self)["fmt_locale"] = locale


UnitRegistry.set_fmt_locale = _ureg_set_fmt_locale  # type: ignore[attr-defined]

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
