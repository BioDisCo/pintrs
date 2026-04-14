"""NumPy array quantity support for pintrs.

Provides ArrayQuantity that wraps numpy arrays with units, implementing
__array_ufunc__ for transparent numpy integration.
"""

# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportInvalidTypeForm=false, reportOptionalCall=false, reportPossiblyUnboundVariable=false, reportPrivateUsage=false, reportRedeclaration=false, reportReturnType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnnecessaryIsInstance=false, reportUnusedFunction=false

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Literal, TypeVar, cast

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np
    from numpy.typing import DTypeLike, NDArray

    from pintrs._core import Quantity as ScalarQuantity
    from pintrs._core import Unit
    from pintrs._core import UnitRegistry as _UnitRegistry

try:
    import numpy as np

    has_numpy = True
except ImportError:
    has_numpy = False

if has_numpy:
    from pintrs._core import Quantity as ScalarQuantity
    from pintrs._core import Unit as _Unit
    from pintrs._core import UnitRegistry as _UnitRegistry

    try:
        _RustArrayQuantity: Any = getattr(
            __import__("pintrs._core", fromlist=["RustArrayQuantity"]),
            "RustArrayQuantity",
            None,
        )
    except ImportError:
        _RustArrayQuantity = None

    HANDLED_FUNCTIONS: dict[Any, Any] = {}

    _F = TypeVar("_F", bound=Callable[..., Any])

    def implements(numpy_function: Any) -> Callable[[_F], _F]:
        """Decorator registering an __array_function__ handler."""

        def decorator(func: _F) -> _F:
            HANDLED_FUNCTIONS[numpy_function] = func
            return func

        return decorator

    def _mag_of(x: Any) -> Any:
        if isinstance(x, ScalarQuantity):
            return x.magnitude
        if _RustArrayQuantity is not None and isinstance(x, _RustArrayQuantity):
            return np.asarray(x.m)
        try:
            if isinstance(x, ArrayQuantity):
                return x._magnitude
        except NameError:
            pass
        return x

    def _unit_str_of(x: Any) -> str | None:
        if isinstance(x, ScalarQuantity):
            return str(x.units)
        if _RustArrayQuantity is not None and isinstance(x, _RustArrayQuantity):
            return x._units_str  # type: ignore[no-any-return]
        try:
            if isinstance(x, ArrayQuantity):
                return x._units_str
        except NameError:
            pass
        return None

    def _registry_of(x: Any) -> Any:
        if isinstance(x, ScalarQuantity):
            return x._registry  # type: ignore[attr-defined]
        if _RustArrayQuantity is not None and isinstance(x, _RustArrayQuantity):
            return x._registry
        try:
            if isinstance(x, ArrayQuantity):
                return x._registry
        except NameError:
            pass
        return None

    def _first_registry(*candidates: Any) -> Any:
        for c in candidates:
            r = _registry_of(c)
            if r is not None:
                return r
            if isinstance(c, (list, tuple)):
                for item in c:
                    r = _registry_of(item)
                    if r is not None:
                        return r
        return None

    def _unit_obj_of(x: Any) -> Any:
        if isinstance(x, ScalarQuantity):
            return x.units
        if _RustArrayQuantity is not None and isinstance(x, _RustArrayQuantity):
            return x.units
        try:
            if isinstance(x, ArrayQuantity):
                return x._unit_obj
        except NameError:
            pass
        return None

    def _convert_mag(
        mag: Any, src_unit: str | None, dst_unit: str, registry: Any
    ) -> Any:
        if src_unit is None or src_unit == dst_unit or registry is None:
            return mag
        factor = registry._get_conversion_factor(src_unit, dst_unit)
        return mag * factor

    def _make_result(mag: Any, unit_str: str, registry: Any) -> Any:
        if isinstance(mag, np.ndarray):
            return ArrayQuantity(mag, unit_str, registry)
        if np.isscalar(mag) or isinstance(mag, np.generic):
            val = mag.item() if isinstance(mag, np.generic) else mag
            if isinstance(val, (int, float)):
                return registry.Quantity(float(val), unit_str)
            if isinstance(val, complex):
                return registry.Quantity(val, unit_str)
        return mag

    def _dispatch_array_function(
        func: Any,
        types: tuple[type[Any], ...],  # noqa: ARG001
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        handler = HANDLED_FUNCTIONS.get(func)
        if handler is None:
            return NotImplemented
        return handler(*args, **kwargs)

    class ArrayQuantity:
        """A Quantity whose magnitude is a numpy array."""

        __hash__ = None  # type: ignore[assignment]  # mutable, unhashable

        def __init__(
            self,
            magnitude: NDArray[Any] | list[float],
            units: str,
            registry: _UnitRegistry | None = None,
        ) -> None:
            self._magnitude: NDArray[Any] = np.asarray(magnitude)
            self._units_str = units
            self._registry: _UnitRegistry = registry or _UnitRegistry()
            self._unit_obj: Unit = self._registry.Unit(units)

        @property
        def magnitude(self) -> NDArray[Any]:
            return self._magnitude

        @property
        def m(self) -> NDArray[Any]:
            return self._magnitude

        @property
        def units(self) -> Unit:
            return self._unit_obj

        @property
        def units_str(self) -> str:
            return self._units_str

        @property
        def u(self) -> Unit:
            return self._unit_obj

        @property
        def dimensionality(self) -> Any:
            return self._unit_obj.dimensionality

        @property
        def dimensionless(self) -> bool:
            return bool(self._unit_obj.dimensionless)

        @property
        def unitless(self) -> bool:
            return bool(self._unit_obj.dimensionless)

        @property
        def _REGISTRY(self) -> _UnitRegistry:  # noqa: N802
            return self._registry

        def _repr_inline_(self, max_width: int) -> str:
            """Inline representation for xarray display (pint-xarray compat)."""
            unit_str = f"[{self._units_str}]"
            values = [f"{v}" for v in self._magnitude.flat]
            full = f"{unit_str} {' '.join(values)}"
            if len(full) <= max_width:
                return full
            # Try "prefix ... last"
            last = values[-1] if values else ""
            for n in range(len(values) - 1, 0, -1):
                prefix = " ".join(values[:n])
                candidate = f"{unit_str} {prefix} ... {last}"
                if len(candidate) <= max_width:
                    return candidate
            # Just first value with ...
            if values:
                candidate = f"{unit_str} {values[0]}..."
                if len(candidate) <= max_width:
                    return candidate
            return f"{unit_str} ..."

        @property
        def shape(self) -> tuple[int, ...]:
            result: tuple[int, ...] = self._magnitude.shape
            return result

        @property
        def ndim(self) -> int:
            return int(self._magnitude.ndim)

        @property
        def dtype(self) -> np.dtype[Any]:
            result: np.dtype[Any] = self._magnitude.dtype
            return result

        @property
        def T(self) -> ArrayQuantity:  # noqa: N802
            return ArrayQuantity(
                self._magnitude.T,
                self._units_str,
                self._registry,
            )

        @property
        def flat(self) -> np.flatiter[Any]:
            result: np.flatiter[Any] = self._magnitude.flat
            return result

        @property
        def real(self) -> ArrayQuantity:
            return ArrayQuantity(
                np.real(self._magnitude),
                self._units_str,
                self._registry,
            )

        @property
        def imag(self) -> ArrayQuantity:
            return ArrayQuantity(
                np.imag(self._magnitude),
                self._units_str,
                self._registry,
            )

        @staticmethod
        def _coerce_units(units: Any) -> str:
            """Convert a str, Unit, or Quantity to a unit string."""
            if isinstance(units, str):
                return units
            return str(getattr(units, "units", units))

        def m_as(self, units: Any) -> NDArray[Any]:
            u = self._coerce_units(units)
            factor = self._registry._get_conversion_factor(
                self._units_str,
                u,
            )
            return self._magnitude * factor

        def to(self, units: Any, *contexts: Any) -> ArrayQuantity:
            u = self._coerce_units(units)
            factor = self._registry._get_conversion_factor(
                self._units_str,
                u,
            )
            return ArrayQuantity(
                self._magnitude * factor,
                u,
                self._registry,
            )

        def ito(self, units: Any) -> None:
            u = self._coerce_units(units)
            factor = self._registry._get_conversion_factor(
                self._units_str,
                u,
            )
            self._magnitude = self._magnitude * factor
            self._units_str = u
            self._unit_obj = self._registry.Unit(u)

        def to_base_units(self) -> ArrayQuantity:
            factor, unit_str = self._registry._get_root_units(
                self._units_str,
            )
            return ArrayQuantity(
                self._magnitude * factor,
                unit_str,
                self._registry,
            )

        def to_root_units(self) -> ArrayQuantity:
            return self.to_base_units()

        def to_compact(self) -> ArrayQuantity:
            scalar_q = self._registry.Quantity(
                float(np.mean(np.abs(self._magnitude))),
                self._units_str,
            )
            compact = scalar_q.to_compact()
            target = str(compact.units)
            return self.to(target)

        def is_compatible_with(
            self,
            other: str | ArrayQuantity | ScalarQuantity,
        ) -> bool:
            scalar = self._registry.Quantity(1.0, self._units_str)
            if isinstance(other, str):
                return bool(scalar.is_compatible_with(other))
            if isinstance(other, ArrayQuantity):
                return bool(scalar.is_compatible_with(other._units_str))
            return bool(scalar.is_compatible_with(other))

        def check(self, dimension: Any) -> bool:
            scalar = self._registry.Quantity(1.0, self._units_str)
            if isinstance(dimension, str):
                return bool(scalar.check(dimension))
            return bool(scalar.is_compatible_with(dimension))

        def clip(
            self,
            min: float | None = None,  # noqa: A002
            max: float | None = None,  # noqa: A002
        ) -> ArrayQuantity:
            return ArrayQuantity(
                self._magnitude.clip(cast("float", min), cast("float", max)),
                self._units_str,
                self._registry,
            )

        def sum(self, axis: int | None = None) -> ArrayQuantity | ScalarQuantity:
            result = self._magnitude.sum(axis=axis)
            if np.ndim(result) == 0:
                return self._registry.Quantity(float(result), self._units_str)
            return ArrayQuantity(result, self._units_str, self._registry)

        def mean(self, axis: int | None = None) -> ArrayQuantity | ScalarQuantity:
            result = self._magnitude.mean(axis=axis)
            if np.ndim(result) == 0:
                return self._registry.Quantity(float(result), self._units_str)
            return ArrayQuantity(result, self._units_str, self._registry)

        def reshape(self, *shape: Any) -> ArrayQuantity:
            if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
                shape = tuple(shape[0])
            return ArrayQuantity(
                self._magnitude.reshape(shape),
                self._units_str,
                self._registry,
            )

        def flatten(self, order: str = "C") -> ArrayQuantity:
            return ArrayQuantity(
                self._magnitude.flatten(order=order),
                self._units_str,
                self._registry,
            )

        def ravel(self, order: str = "C") -> ArrayQuantity:
            return ArrayQuantity(
                self._magnitude.ravel(order=order),
                self._units_str,
                self._registry,
            )

        def squeeze(self, axis: Any = None) -> ArrayQuantity:
            return ArrayQuantity(
                self._magnitude.squeeze(axis=axis),
                self._units_str,
                self._registry,
            )

        def transpose(self, *axes: Any) -> ArrayQuantity:
            return ArrayQuantity(
                self._magnitude.transpose(*axes),
                self._units_str,
                self._registry,
            )

        def prod(self, axis: int | None = None) -> Any:
            return self._magnitude.prod(axis=axis)

        def tolist(self) -> list[float]:
            return list(self._magnitude.tolist())

        def fill(
            self,
            value: float | ScalarQuantity | ArrayQuantity,
        ) -> None:
            if isinstance(value, (ScalarQuantity, ArrayQuantity)):
                self._magnitude.fill(value.magnitude)
            else:
                self._magnitude.fill(value)

        def put(
            self,
            indices: NDArray[np.intp],
            values: float | ScalarQuantity | ArrayQuantity,
        ) -> None:
            if isinstance(values, (ScalarQuantity, ArrayQuantity)):
                self._magnitude.put(indices, values.magnitude)
            else:
                self._magnitude.put(indices, values)

        def searchsorted(
            self,
            v: float | ScalarQuantity | ArrayQuantity,
            side: Literal["left", "right"] = "left",
        ) -> Any:
            mag = v.magnitude if isinstance(v, (ScalarQuantity, ArrayQuantity)) else v
            return self._magnitude.searchsorted(mag, side=side)  # pyright: ignore[reportCallIssue]

        def dot(
            self,
            other: ArrayQuantity | NDArray[Any],
        ) -> Any:
            mag = other._magnitude if isinstance(other, ArrayQuantity) else other
            return self._magnitude.dot(mag)

        def copy(self) -> ArrayQuantity:
            """Return a copy of this quantity."""
            return ArrayQuantity(
                self._magnitude.copy(),
                self._units_str,
                self._registry,
            )

        def __len__(self) -> int:
            return len(self._magnitude)

        def __getitem__(
            self,
            key: int | slice | NDArray[np.intp],
        ) -> ArrayQuantity | ScalarQuantity:
            result = self._magnitude[key]
            if isinstance(result, np.ndarray):
                return ArrayQuantity(
                    result,
                    self._units_str,
                    self._registry,
                )
            return self._registry.Quantity(float(result), self._units_str)

        def __setitem__(
            self,
            key: int | slice | NDArray[np.intp],
            value: float | ScalarQuantity | ArrayQuantity,
        ) -> None:
            if hasattr(value, "magnitude"):
                converted = value.to(self._units_str)  # type: ignore[union-attr]
                self._magnitude[key] = converted.magnitude
            else:
                self._magnitude[key] = value

        def __iter__(self) -> Iterator[ArrayQuantity | ScalarQuantity]:
            for val in self._magnitude:
                if isinstance(val, np.ndarray):
                    yield ArrayQuantity(
                        val,
                        self._units_str,
                        self._registry,
                    )
                else:
                    yield self._registry.Quantity(
                        float(val),
                        self._units_str,
                    )

        def __repr__(self) -> str:
            return f"<Quantity({self._magnitude}, '{self._units_str}')>"

        def __str__(self) -> str:
            return f"{self._magnitude} {self._units_str}"

        def __neg__(self) -> ArrayQuantity:
            return ArrayQuantity(
                -self._magnitude,
                self._units_str,
                self._registry,
            )

        def __abs__(self) -> ArrayQuantity:
            return ArrayQuantity(
                np.abs(self._magnitude),
                self._units_str,
                self._registry,
            )

        def _get_other_mag_and_units(
            self,
            other: ArrayQuantity | ScalarQuantity | float,
        ) -> tuple[NDArray[Any] | float, str | None]:
            if isinstance(other, ArrayQuantity):
                return other._magnitude, other._units_str
            if isinstance(other, ScalarQuantity):
                return other.magnitude, str(other.units)
            if isinstance(other, _Unit):
                return 1.0, str(other)
            return other, None

        def __add__(
            self,
            other: ArrayQuantity | ScalarQuantity | float,
        ) -> ArrayQuantity:
            mag, units = self._get_other_mag_and_units(other)
            if units is not None and units != self._units_str:
                factor = self._registry._get_conversion_factor(
                    units,
                    self._units_str,
                )
                mag = mag * factor
            return ArrayQuantity(
                self._magnitude + mag,
                self._units_str,
                self._registry,
            )

        def __radd__(
            self,
            other: ArrayQuantity | ScalarQuantity | float,
        ) -> ArrayQuantity:
            return self.__add__(other)

        def __sub__(
            self,
            other: ArrayQuantity | ScalarQuantity | float,
        ) -> ArrayQuantity:
            mag, units = self._get_other_mag_and_units(other)
            if units is not None and units != self._units_str:
                factor = self._registry._get_conversion_factor(
                    units,
                    self._units_str,
                )
                mag = mag * factor
            return ArrayQuantity(
                self._magnitude - mag,
                self._units_str,
                self._registry,
            )

        def __rsub__(
            self,
            other: ArrayQuantity | ScalarQuantity | float,
        ) -> ArrayQuantity:
            result = self.__sub__(other)
            return ArrayQuantity(
                -result._magnitude,
                result._units_str,
                result._registry,
            )

        def __mul__(
            self,
            other: ArrayQuantity | ScalarQuantity | float,
        ) -> ArrayQuantity:
            mag, units = self._get_other_mag_and_units(other)
            if units is not None:
                new_units = str(
                    self._unit_obj * self._registry.Unit(units),
                )
            else:
                new_units = self._units_str
            return ArrayQuantity(
                self._magnitude * mag,
                new_units,
                self._registry,
            )

        def __rmul__(
            self,
            other: ArrayQuantity | ScalarQuantity | float,
        ) -> ArrayQuantity:
            return self.__mul__(other)

        def __truediv__(
            self,
            other: ArrayQuantity | ScalarQuantity | float,
        ) -> ArrayQuantity:
            mag, units = self._get_other_mag_and_units(other)
            if units is not None:
                new_units = str(
                    self._unit_obj / self._registry.Unit(units),
                )
            else:
                new_units = self._units_str
            return ArrayQuantity(
                self._magnitude / mag,
                new_units,
                self._registry,
            )

        def __rtruediv__(
            self,
            other: ArrayQuantity | ScalarQuantity | float,
        ) -> ArrayQuantity:
            mag, units = self._get_other_mag_and_units(other)
            if units is not None:
                new_units = str(
                    self._registry.Unit(units) / self._unit_obj,
                )
            else:
                new_units = str(
                    self._registry.Unit("dimensionless") / self._unit_obj,
                )
            return ArrayQuantity(
                mag / self._magnitude,
                new_units,
                self._registry,
            )

        def __pow__(self, exp: float) -> ArrayQuantity:
            new_units = str(self._unit_obj**exp)
            return ArrayQuantity(
                self._magnitude**exp,
                new_units,
                self._registry,
            )

        def __eq__(self, other: object) -> bool:
            if isinstance(other, ArrayQuantity):
                if other._units_str != self._units_str:
                    factor = self._registry._get_conversion_factor(
                        other._units_str,
                        self._units_str,
                    )
                    return bool(
                        np.array_equal(self._magnitude, other._magnitude * factor),
                    )
                return bool(np.array_equal(self._magnitude, other._magnitude))
            if isinstance(other, ScalarQuantity):
                return bool(np.array_equal(self._magnitude, other.magnitude))
            if isinstance(other, (np.ndarray, list, tuple, int, float, complex)):
                return bool(np.array_equal(self._magnitude, other))
            return False

        def __ne__(self, other: object) -> bool:
            return not self.__eq__(other)

        def __lt__(
            self,
            other: ArrayQuantity | ScalarQuantity | float,
        ) -> Any:
            mag, units = self._get_other_mag_and_units(other)
            if units is not None and units != self._units_str:
                factor = self._registry._get_conversion_factor(
                    units,
                    self._units_str,
                )
                mag = mag * factor
            return self._magnitude < mag

        def __le__(
            self,
            other: ArrayQuantity | ScalarQuantity | float,
        ) -> Any:
            mag, units = self._get_other_mag_and_units(other)
            if units is not None and units != self._units_str:
                factor = self._registry._get_conversion_factor(
                    units,
                    self._units_str,
                )
                mag = mag * factor
            return self._magnitude <= mag

        def __gt__(
            self,
            other: ArrayQuantity | ScalarQuantity | float,
        ) -> Any:
            mag, units = self._get_other_mag_and_units(other)
            if units is not None and units != self._units_str:
                factor = self._registry._get_conversion_factor(
                    units,
                    self._units_str,
                )
                mag = mag * factor
            return self._magnitude > mag

        def __ge__(
            self,
            other: ArrayQuantity | ScalarQuantity | float,
        ) -> Any:
            mag, units = self._get_other_mag_and_units(other)
            if units is not None and units != self._units_str:
                factor = self._registry._get_conversion_factor(
                    units,
                    self._units_str,
                )
                mag = mag * factor
            return self._magnitude >= mag

        def __bool__(self) -> bool:
            return bool(np.any(self._magnitude != 0))

        def __array__(
            self,
            dtype: DTypeLike | None = None,
        ) -> NDArray[Any]:
            if dtype is not None:
                return np.asarray(self._magnitude, dtype=dtype)
            return np.asarray(self._magnitude)

        def astype(
            self,
            dtype: DTypeLike,
            *,
            copy: bool = True,
        ) -> ArrayQuantity:
            new_mag = self._magnitude.astype(dtype, copy=copy)
            return ArrayQuantity(new_mag, self._units_str, self._registry)

        # Ufuncs that require dimensionless input
        _DIMENSIONLESS_UFUNCS: frozenset[np.ufunc] = frozenset()
        # Trig ufuncs that accept radians/degrees/dimensionless
        _TRIG_UFUNCS: frozenset[np.ufunc] = frozenset()
        # Inverse trig ufuncs that return radians
        _INVERSE_TRIG_UFUNCS: frozenset[np.ufunc] = frozenset()
        # arctan2: two args with matching units, returns radians
        _ARCTAN2_UFUNCS: frozenset[np.ufunc] = frozenset()
        # Ufuncs that require matching dimensions for two inputs
        _MATCHING_DIMS_UFUNCS: frozenset[np.ufunc] = frozenset()

        @staticmethod
        def _ufunc_output_units(
            ufunc: np.ufunc,
            input_units: list[Unit | None],
            processed: list[NDArray[Any] | float],
            default_units: str,
        ) -> str:
            """Compute output units for a numpy ufunc."""
            if ufunc in (np.multiply, np.matmul):
                u0 = input_units[0] if input_units else None
                u1 = input_units[1] if len(input_units) > 1 else None
                if u0 is not None and u1 is not None:
                    return str(u0 * u1)
                if u0 is not None:
                    return str(u0)
                return str(u1) if u1 is not None else default_units
            if ufunc in (np.divide, np.true_divide, np.floor_divide):
                u0 = input_units[0] if input_units else None
                u1 = input_units[1] if len(input_units) > 1 else None
                if u0 is not None and u1 is not None:
                    return str(u0 / u1)
                return str(u0) if u0 is not None else default_units
            if ufunc is np.power:
                exp = processed[1] if len(processed) > 1 else 1
                if isinstance(exp, np.ndarray):
                    exp = float(exp.flat[0])
                u0 = input_units[0]
                return str(u0 ** float(exp)) if u0 is not None else default_units
            power_map: dict[np.ufunc, float] = {
                np.sqrt: 0.5,
                np.square: 2.0,
                np.cbrt: 1.0 / 3.0,
                np.reciprocal: -1.0,
            }
            if ufunc in power_map:
                u0 = input_units[0]
                return str(u0 ** power_map[ufunc]) if u0 is not None else default_units
            return default_units

        def _check_additive_compat(
            self,
            ufunc: np.ufunc,
            inputs: tuple[Any, ...],
            processed: list[NDArray[Any] | float],
            input_units: list[Unit | None],
        ) -> list[NDArray[Any] | float]:
            """Check dimensionality and convert units for add/subtract."""
            from pintrs import DimensionalityError  # noqa: PLC0415

            if ufunc not in self._MATCHING_DIMS_UFUNCS:
                return processed
            units_with_idx = [
                (i, u) for i, u in enumerate(input_units) if u is not None
            ]
            if len(units_with_idx) < 2:
                return processed
            _, base_unit = units_with_idx[0]
            result = list(processed)
            for idx, unit in units_with_idx[1:]:
                if str(unit.dimensionality) != str(base_unit.dimensionality):
                    msg = f"Cannot convert from '{unit}' to '{base_unit}'"
                    raise DimensionalityError(msg)
                if str(unit) != str(base_unit):
                    inp = inputs[idx]
                    if isinstance(inp, ArrayQuantity):
                        converted = inp.to(str(base_unit))
                        result[idx] = converted._magnitude
                    elif isinstance(inp, ScalarQuantity):
                        converted_q = inp.to(str(base_unit))
                        result[idx] = converted_q.magnitude
            return result

        def _check_trig_units(
            self,
            ufunc: np.ufunc,
            inputs: tuple[Any, ...],
            processed: list[NDArray[Any] | float],
            input_units: list[Unit | None],
        ) -> list[NDArray[Any] | float]:
            """Validate and convert units for trig/exp/log ufuncs."""
            from pintrs import DimensionalityError  # noqa: PLC0415

            if ufunc in self._TRIG_UFUNCS:
                u = input_units[0] if input_units else None
                if u is not None:
                    u_str = str(u)
                    if u_str in ("rad", "radian"):
                        return processed
                    if u_str in ("deg", "degree"):
                        result = list(processed)
                        result[0] = np.deg2rad(processed[0])
                        return result
                    q_check = self._registry.Quantity(1.0, u_str)
                    if q_check.dimensionless:
                        return processed
                    msg = (
                        f"Cannot apply '{ufunc.__name__}' to quantity "
                        f"with units '{u_str}'"
                    )
                    raise DimensionalityError(msg)
            if ufunc in self._INVERSE_TRIG_UFUNCS:
                u = input_units[0] if input_units else None
                if u is not None:
                    q_check = self._registry.Quantity(1.0, str(u))
                    if not q_check.dimensionless:
                        msg = (
                            f"Cannot apply '{ufunc.__name__}' to quantity "
                            f"with units '{u}'"
                        )
                        raise DimensionalityError(msg)
            if ufunc in self._DIMENSIONLESS_UFUNCS:
                u = input_units[0] if input_units else None
                if u is not None:
                    q_check = self._registry.Quantity(1.0, str(u))
                    if not q_check.dimensionless:
                        msg = (
                            f"Cannot apply '{ufunc.__name__}' to quantity "
                            f"with units '{u}'"
                        )
                        raise DimensionalityError(msg)
            return processed

        _COMPARISON_UFUNCS: frozenset[np.ufunc] = frozenset()

        def __array_ufunc__(
            self,
            ufunc: np.ufunc,
            method: str,
            *inputs: ArrayQuantity | ScalarQuantity | float,
            **kwargs: object,
        ) -> Any:
            """Support numpy ufuncs with unit tracking."""
            processed: list[NDArray[Any] | float] = []
            input_units: list[Unit | None] = []
            for inp in inputs:
                if isinstance(inp, ArrayQuantity):
                    processed.append(inp._magnitude)
                    input_units.append(inp._unit_obj)
                elif isinstance(inp, ScalarQuantity):
                    processed.append(inp.magnitude)
                    input_units.append(inp.units)
                else:
                    processed.append(inp)
                    input_units.append(None)

            processed = self._check_additive_compat(
                ufunc,
                inputs,
                processed,
                input_units,
            )
            processed = self._check_trig_units(
                ufunc,
                inputs,
                processed,
                input_units,
            )

            result = getattr(ufunc, method)(*processed, **kwargs)

            if ufunc in self._COMPARISON_UFUNCS:
                return result

            if ufunc in self._TRIG_UFUNCS or ufunc in self._DIMENSIONLESS_UFUNCS:
                out_units = "dimensionless"
            elif ufunc in self._INVERSE_TRIG_UFUNCS or ufunc in self._ARCTAN2_UFUNCS:
                out_units = "radian"
            else:
                out_units = self._ufunc_output_units(
                    ufunc,
                    input_units,
                    processed,
                    self._units_str,
                )

            if isinstance(result, np.ndarray):
                return ArrayQuantity(result, out_units, self._registry)
            if np.isscalar(result):
                magnitude = result.item() if isinstance(result, np.generic) else result
                if isinstance(magnitude, (int, float)):
                    return self._registry.Quantity(float(magnitude), out_units)
                if isinstance(magnitude, str):
                    return self._registry.Quantity(magnitude, out_units)
            return result

        def __array_function__(
            self,
            func: Any,
            types: tuple[type[Any], ...],
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
        ) -> Any:
            return _dispatch_array_function(func, types, args, kwargs)

    ArrayQuantity._COMPARISON_UFUNCS = frozenset(
        {
            np.equal,
            np.not_equal,
            np.less,
            np.less_equal,
            np.greater,
            np.greater_equal,
            np.isnan,
            np.isinf,
            np.isfinite,
            np.signbit,
            np.logical_and,
            np.logical_or,
            np.logical_not,
        },
    )

    ArrayQuantity._DIMENSIONLESS_UFUNCS = frozenset(
        {
            np.exp,
            np.exp2,
            np.expm1,
            np.log,
            np.log2,
            np.log10,
            np.log1p,
            np.logaddexp,
            np.logaddexp2,
        },
    )

    ArrayQuantity._TRIG_UFUNCS = frozenset(
        {
            np.sin,
            np.cos,
            np.tan,
            np.sinh,
            np.cosh,
            np.tanh,
        },
    )

    ArrayQuantity._INVERSE_TRIG_UFUNCS = frozenset(
        {
            np.arcsin,
            np.arccos,
            np.arctan,
            np.arcsinh,
            np.arccosh,
            np.arctanh,
        },
    )

    # arctan2 takes two args with matching units and returns radians
    ArrayQuantity._ARCTAN2_UFUNCS = frozenset({np.arctan2})

    ArrayQuantity._MATCHING_DIMS_UFUNCS = frozenset(
        {
            np.add,
            np.subtract,
            np.remainder,
            np.mod,
            np.fmod,
            np.copysign,
            np.nextafter,
            np.greater,
            np.greater_equal,
            np.less,
            np.less_equal,
            np.equal,
            np.not_equal,
            np.maximum,
            np.minimum,
            np.hypot,
            np.arctan2,
        },
    )

    # ------------------------------------------------------------------
    # __array_function__ handlers
    #
    # These are dispatched for ArrayQuantity (Python), RustArrayQuantity,
    # and Quantity (scalar) inputs uniformly.
    # ------------------------------------------------------------------

    @implements(np.linspace)
    def _linspace_impl(  # noqa: PLR0913
        start: Any,
        stop: Any,
        num: int = 50,
        endpoint: bool = True,
        retstep: bool = False,
        dtype: Any = None,
        axis: int = 0,
    ) -> Any:
        reg = _first_registry(start, stop)
        start_u = _unit_str_of(start)
        stop_u = _unit_str_of(stop)
        out_unit = start_u or stop_u
        linspace_fn: Any = np.linspace
        if out_unit is None:
            return linspace_fn(
                start,
                stop,
                num=num,
                endpoint=endpoint,
                retstep=retstep,
                dtype=dtype,
                axis=axis,
            )
        start_mag = _convert_mag(_mag_of(start), start_u, out_unit, reg)
        stop_mag = _convert_mag(_mag_of(stop), stop_u, out_unit, reg)
        result = linspace_fn(
            start_mag,
            stop_mag,
            num=num,
            endpoint=endpoint,
            retstep=retstep,
            dtype=dtype,
            axis=axis,
        )
        if retstep:
            arr, step = result
            return (
                _make_result(arr, out_unit, reg),
                _make_result(step, out_unit, reg),
            )
        return _make_result(result, out_unit, reg)

    @implements(np.logspace)
    def _logspace_impl(  # noqa: PLR0913
        start: Any,
        stop: Any,
        num: int = 50,
        endpoint: bool = True,
        base: float = 10.0,
        dtype: Any = None,
        axis: int = 0,
    ) -> Any:
        # logspace inputs must be dimensionless; any Quantity output needs
        # its own unit set separately by the caller.
        start_mag = _mag_of(start)
        stop_mag = _mag_of(stop)
        return np.logspace(start_mag, stop_mag, num, endpoint, base, dtype, axis)

    @implements(np.geomspace)
    def _geomspace_impl(  # noqa: PLR0913
        start: Any,
        stop: Any,
        num: int = 50,
        endpoint: bool = True,
        dtype: Any = None,
        axis: int = 0,
    ) -> Any:
        reg = _first_registry(start, stop)
        start_u = _unit_str_of(start)
        stop_u = _unit_str_of(stop)
        out_unit = start_u or stop_u
        if out_unit is None:
            return np.geomspace(start, stop, num, endpoint, dtype, axis)
        start_mag = _convert_mag(_mag_of(start), start_u, out_unit, reg)
        stop_mag = _convert_mag(_mag_of(stop), stop_u, out_unit, reg)
        result = np.geomspace(start_mag, stop_mag, num, endpoint, dtype, axis)
        return _make_result(result, out_unit, reg)

    @implements(np.full)
    def _full_impl(
        shape: Any,
        fill_value: Any,
        dtype: Any = None,
        order: str = "C",
    ) -> Any:
        reg = _registry_of(fill_value)
        unit = _unit_str_of(fill_value)
        mag = _mag_of(fill_value)
        result = np.full(shape, mag, dtype=dtype, order=order)
        if unit is None:
            return result
        return _make_result(result, unit, reg)

    @implements(np.full_like)
    def _full_like_impl(  # noqa: PLR0913
        a: Any,
        fill_value: Any,
        dtype: Any = None,
        order: str = "K",
        subok: bool = True,
        shape: Any = None,
    ) -> Any:
        reg = _first_registry(a, fill_value)
        a_u = _unit_str_of(a)
        fill_u = _unit_str_of(fill_value)
        unit = a_u or fill_u
        mag = _mag_of(a)
        fill_mag = _mag_of(fill_value)
        if fill_u is not None and a_u is not None and reg is not None:
            fill_mag = _convert_mag(fill_mag, fill_u, a_u, reg)
        result = np.full_like(
            mag, fill_mag, dtype=dtype, order=order, subok=subok, shape=shape
        )
        if unit is None:
            return result
        return _make_result(result, unit, reg)

    @implements(np.zeros_like)
    def _zeros_like_impl(
        a: Any,
        dtype: Any = None,
        order: str = "K",
        subok: bool = True,
        shape: Any = None,
    ) -> Any:
        mag = _mag_of(a)
        unit = _unit_str_of(a)
        reg = _registry_of(a)
        result = np.zeros_like(mag, dtype=dtype, order=order, subok=subok, shape=shape)
        if unit is None:
            return result
        return _make_result(result, unit, reg)

    @implements(np.ones_like)
    def _ones_like_impl(
        a: Any,
        dtype: Any = None,
        order: str = "K",
        subok: bool = True,
        shape: Any = None,
    ) -> Any:
        mag = _mag_of(a)
        unit = _unit_str_of(a)
        reg = _registry_of(a)
        result = np.ones_like(mag, dtype=dtype, order=order, subok=subok, shape=shape)
        if unit is None:
            return result
        return _make_result(result, unit, reg)

    @implements(np.empty_like)
    def _empty_like_impl(
        prototype: Any,
        dtype: Any = None,
        order: str = "K",
        subok: bool = True,
        shape: Any = None,
    ) -> Any:
        mag = _mag_of(prototype)
        unit = _unit_str_of(prototype)
        reg = _registry_of(prototype)
        result = np.empty_like(mag, dtype=dtype, order=order, subok=subok, shape=shape)
        if unit is None:
            return result
        return _make_result(result, unit, reg)

    @implements(np.broadcast_to)
    def _broadcast_to_impl(
        array: Any,
        shape: Any,
        subok: bool = False,
    ) -> Any:
        mag = _mag_of(array)
        unit = _unit_str_of(array)
        reg = _registry_of(array)
        result = np.broadcast_to(mag, shape, subok=subok)
        if unit is None:
            return result
        return _make_result(result, unit, reg)

    @implements(np.append)
    def _append_impl(
        arr: Any,
        values: Any,
        axis: int | None = None,
    ) -> Any:
        reg = _first_registry(arr, values)
        arr_u = _unit_str_of(arr)
        val_u = _unit_str_of(values)
        out_unit = arr_u or val_u
        arr_mag = _convert_mag(_mag_of(arr), arr_u, out_unit or "", reg)
        val_mag = _convert_mag(_mag_of(values), val_u, out_unit or "", reg)
        result = np.append(arr_mag, val_mag, axis=axis)
        if out_unit is None:
            return result
        return _make_result(result, out_unit, reg)

    @implements(np.concatenate)
    def _concatenate_impl(
        arrays: Any,
        axis: int | None = 0,
        out: Any = None,
        dtype: Any = None,
        casting: str = "same_kind",
    ) -> Any:
        reg = _first_registry(*arrays)
        out_unit: str | None = None
        for a in arrays:
            u = _unit_str_of(a)
            if u is not None:
                out_unit = u
                break
        mags = [
            _convert_mag(_mag_of(a), _unit_str_of(a), out_unit or "", reg)
            for a in arrays
        ]
        result = np.concatenate(
            mags,
            axis=axis,
            out=out,
            dtype=dtype,
            casting=casting,
        )
        if out_unit is None:
            return result
        return _make_result(result, out_unit, reg)

    @implements(np.stack)
    def _stack_impl(
        arrays: Any,
        axis: int = 0,
        out: Any = None,
    ) -> Any:
        reg = _first_registry(*arrays)
        out_unit: str | None = None
        for a in arrays:
            u = _unit_str_of(a)
            if u is not None:
                out_unit = u
                break
        mags = [
            _convert_mag(_mag_of(a), _unit_str_of(a), out_unit or "", reg)
            for a in arrays
        ]
        result = np.stack(mags, axis=axis, out=out)
        if out_unit is None:
            return result
        return _make_result(result, out_unit, reg)

    @implements(np.where)
    def _where_impl(
        condition: Any,
        x: Any = None,
        y: Any = None,
    ) -> Any:
        if x is None and y is None:
            return np.where(condition)
        reg = _first_registry(x, y)
        x_u = _unit_str_of(x)
        y_u = _unit_str_of(y)
        out_unit = x_u or y_u
        x_mag = _convert_mag(_mag_of(x), x_u, out_unit or "", reg)
        y_mag = _convert_mag(_mag_of(y), y_u, out_unit or "", reg)
        result = np.where(condition, x_mag, y_mag)
        if out_unit is None:
            return result
        return _make_result(result, out_unit, reg)

    @implements(np.clip)
    def _clip_impl(
        a: Any,
        a_min: Any = None,
        a_max: Any = None,
        out: Any = None,
        **kwargs: Any,
    ) -> Any:
        reg = _first_registry(a, a_min, a_max)
        unit = _unit_str_of(a)
        a_mag = _mag_of(a)
        a_min_mag = (
            _convert_mag(_mag_of(a_min), _unit_str_of(a_min), unit or "", reg)
            if a_min is not None
            else None
        )
        a_max_mag = (
            _convert_mag(_mag_of(a_max), _unit_str_of(a_max), unit or "", reg)
            if a_max is not None
            else None
        )
        result = np.clip(a_mag, a_min_mag, a_max_mag, out=out, **kwargs)
        if unit is None:
            return result
        return _make_result(result, unit, reg)

    @implements(np.reshape)
    def _reshape_impl(a: Any, *args: Any, **kwargs: Any) -> Any:
        mag = _mag_of(a)
        unit = _unit_str_of(a)
        reg = _registry_of(a)
        result = np.reshape(mag, *args, **kwargs)
        if unit is None:
            return result
        return _make_result(result, unit, reg)

    def _make_unary_reducer(
        np_func: Any,
        unit_transform: Any = None,
    ) -> Any:
        def _impl(a: Any, *args: Any, **kwargs: Any) -> Any:
            mag = _mag_of(a)
            unit = _unit_str_of(a)
            reg = _registry_of(a)
            result = np_func(mag, *args, **kwargs)
            if unit is None:
                return result
            out_unit = unit_transform(unit, reg) if unit_transform else unit
            return _make_result(result, out_unit, reg)

        return _impl

    _reducer_funcs: tuple[Any, ...] = (
        np.sum,
        np.mean,
        np.median,
        np.min,
        np.amin,
        np.max,
        np.amax,
        np.std,
        np.ptp,
        np.cumsum,
        np.sort,
        np.ravel,
        np.squeeze,
        np.transpose,
        np.atleast_1d,
        np.atleast_2d,
        np.atleast_3d,
        np.flip,
        np.fliplr,
        np.flipud,
        np.roll,
        np.copy,
        np.real,
        np.imag,
    )
    for _reducer_func in _reducer_funcs:
        implements(_reducer_func)(_make_unary_reducer(_reducer_func))

    def _var_unit_transform(unit: str, reg: Any) -> str:
        u = reg.Unit(unit)
        return str(u * u)

    implements(np.var)(_make_unary_reducer(np.var, _var_unit_transform))

    def _make_pass_through(np_func: Any) -> Any:
        def _impl(a: Any, *args: Any, **kwargs: Any) -> Any:
            return np_func(_mag_of(a), *args, **kwargs)

        return _impl

    _pass_through_funcs: tuple[Any, ...] = (
        np.argmin,
        np.argmax,
        np.argsort,
        np.isreal,
        np.iscomplex,
        np.isnan,
        np.isinf,
        np.isfinite,
        np.nonzero,
        np.count_nonzero,
        np.shape,
        np.ndim,
        np.size,
    )
    for _pass_func in _pass_through_funcs:
        implements(_pass_func)(_make_pass_through(_pass_func))

    @implements(np.cumprod)
    def _cumprod_impl(a: Any, *args: Any, **kwargs: Any) -> Any:
        unit = _unit_str_of(a)
        if unit is not None:
            from pintrs import DimensionalityError  # noqa: PLC0415

            reg = _registry_of(a)
            scalar = reg.Quantity(1.0, unit)
            if not scalar.dimensionless:
                msg = "Cannot apply cumprod to a dimensioned quantity"
                raise DimensionalityError(msg)
        return np.cumprod(_mag_of(a), *args, **kwargs)

    @implements(np.percentile)
    def _percentile_impl(a: Any, q: Any, *args: Any, **kwargs: Any) -> Any:
        mag = _mag_of(a)
        unit = _unit_str_of(a)
        reg = _registry_of(a)
        result = np.percentile(mag, q, *args, **kwargs)
        if unit is None:
            return result
        return _make_result(result, unit, reg)

    @implements(np.quantile)
    def _quantile_impl(a: Any, q: Any, *args: Any, **kwargs: Any) -> Any:
        mag = _mag_of(a)
        unit = _unit_str_of(a)
        reg = _registry_of(a)
        result = np.quantile(mag, q, *args, **kwargs)
        if unit is None:
            return result
        return _make_result(result, unit, reg)

    def _binary_product_handler(np_func: Any) -> Any:
        def _impl(a: Any, b: Any, *args: Any, **kwargs: Any) -> Any:
            a_mag = _mag_of(a)
            b_mag = _mag_of(b)
            a_u = _unit_obj_of(a)
            b_u = _unit_obj_of(b)
            reg = _first_registry(a, b)
            result = np_func(a_mag, b_mag, *args, **kwargs)
            if a_u is not None and b_u is not None:
                out_u = str(a_u * b_u)
            elif a_u is not None:
                out_u = str(a_u)
            elif b_u is not None:
                out_u = str(b_u)
            else:
                return result
            return _make_result(result, out_u, reg)

        return _impl

    implements(np.dot)(_binary_product_handler(np.dot))
    implements(np.cross)(_binary_product_handler(np.cross))
    implements(np.outer)(_binary_product_handler(np.outer))
    implements(np.inner)(_binary_product_handler(np.inner))
    implements(np.vdot)(_binary_product_handler(np.vdot))
    implements(np.correlate)(_binary_product_handler(np.correlate))
    implements(np.tensordot)(_binary_product_handler(np.tensordot))
    implements(np.matmul)(_binary_product_handler(np.matmul))

    @implements(np.linalg.norm)
    def _norm_impl(
        x: Any,
        ord: Any = None,  # noqa: A002
        axis: Any = None,
        keepdims: bool = False,
    ) -> Any:
        mag = _mag_of(x)
        unit = _unit_str_of(x)
        reg = _registry_of(x)
        result = np.linalg.norm(mag, ord=ord, axis=axis, keepdims=keepdims)
        if unit is None:
            return result
        return _make_result(result, unit, reg)

    _trapezoid: Any = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    if _trapezoid is not None:
        _trap_fn: Any = _trapezoid

        def _trapezoid_impl(
            y: Any,
            x: Any = None,
            dx: float = 1.0,
            axis: int = -1,
        ) -> Any:
            y_mag = _mag_of(y)
            y_u = _unit_obj_of(y)
            reg = _first_registry(y, x)
            if x is not None:
                x_mag = _mag_of(x)
                x_u = _unit_obj_of(x)
                result = _trap_fn(y_mag, x_mag, axis=axis)
            else:
                x_u = None
                result = _trap_fn(y_mag, dx=dx, axis=axis)
            if y_u is not None and x_u is not None:
                out_u = str(y_u * x_u)
            elif y_u is not None:
                out_u = str(y_u)
            elif x_u is not None:
                out_u = str(x_u)
            else:
                return result
            return _make_result(result, out_u, reg)

        implements(_trapezoid)(_trapezoid_impl)

    # ----- Long-tail handlers -----

    _DIFF_SENTINEL = getattr(np, "_NoValue", object())

    @implements(np.diff)
    def _diff_impl(
        a: Any,
        n: int = 1,
        axis: int = -1,
        prepend: Any = _DIFF_SENTINEL,
        append: Any = _DIFF_SENTINEL,
    ) -> Any:
        mag = _mag_of(a)
        unit = _unit_str_of(a)
        reg = _registry_of(a)
        kwargs: dict[str, Any] = {"n": n, "axis": axis}
        if prepend is not _DIFF_SENTINEL:
            kwargs["prepend"] = _convert_mag(
                _mag_of(prepend), _unit_str_of(prepend), unit or "", reg
            )
        if append is not _DIFF_SENTINEL:
            kwargs["append"] = _convert_mag(
                _mag_of(append), _unit_str_of(append), unit or "", reg
            )
        result = np.diff(mag, **kwargs)
        if unit is None:
            return result
        return _make_result(result, unit, reg)

    @implements(np.ediff1d)
    def _ediff1d_impl(
        ary: Any,
        to_end: Any = None,
        to_begin: Any = None,
    ) -> Any:
        mag = _mag_of(ary)
        unit = _unit_str_of(ary)
        reg = _registry_of(ary)
        if to_end is not None:
            to_end = _convert_mag(
                _mag_of(to_end), _unit_str_of(to_end), unit or "", reg
            )
        if to_begin is not None:
            to_begin = _convert_mag(
                _mag_of(to_begin), _unit_str_of(to_begin), unit or "", reg
            )
        result = np.ediff1d(mag, to_end=to_end, to_begin=to_begin)
        if unit is None:
            return result
        return _make_result(result, unit, reg)

    @implements(np.gradient)
    def _gradient_impl(f: Any, *varargs: Any, **kwargs: Any) -> Any:
        f_mag = _mag_of(f)
        f_unit = _unit_obj_of(f)
        reg = _first_registry(f, *varargs)
        var_mags: list[Any] = []
        var_units: list[Any] = []
        for v in varargs:
            var_mags.append(_mag_of(v))
            var_units.append(_unit_obj_of(v))
        result = np.gradient(f_mag, *var_mags, **kwargs)

        def _out_unit(v_u: Any) -> str:
            if f_unit is None and v_u is None:
                return "dimensionless"
            if f_unit is None:
                return str(v_u**-1)
            if v_u is None:
                return str(f_unit)
            return str(f_unit / v_u)

        if isinstance(result, list):
            if not var_units:
                var_units = [None] * len(result)
            return [
                _make_result(
                    r, _out_unit(var_units[i] if i < len(var_units) else None), reg
                )
                for i, r in enumerate(result)
            ]
        return _make_result(result, _out_unit(var_units[0] if var_units else None), reg)

    @implements(np.interp)
    def _interp_impl(  # noqa: PLR0913
        x: Any,
        xp: Any,
        fp: Any,
        left: Any = None,
        right: Any = None,
        period: Any = None,
    ) -> Any:
        reg = _first_registry(x, xp, fp)
        xp_unit = _unit_str_of(xp)
        x_mag = _convert_mag(_mag_of(x), _unit_str_of(x), xp_unit or "", reg)
        xp_mag = _mag_of(xp)
        fp_mag = _mag_of(fp)
        fp_unit = _unit_str_of(fp)
        left_mag = (
            _convert_mag(_mag_of(left), _unit_str_of(left), fp_unit or "", reg)
            if left is not None
            else None
        )
        right_mag = (
            _convert_mag(_mag_of(right), _unit_str_of(right), fp_unit or "", reg)
            if right is not None
            else None
        )
        period_mag = (
            _convert_mag(_mag_of(period), _unit_str_of(period), xp_unit or "", reg)
            if period is not None
            else None
        )
        result = np.interp(
            x_mag, xp_mag, fp_mag, left=left_mag, right=right_mag, period=period_mag
        )
        if fp_unit is None:
            return result
        return _make_result(result, fp_unit, reg)

    @implements(np.convolve)
    def _convolve_impl(a: Any, v: Any, mode: str = "full") -> Any:
        a_mag = _mag_of(a)
        v_mag = _mag_of(v)
        a_u = _unit_obj_of(a)
        v_u = _unit_obj_of(v)
        reg = _first_registry(a, v)
        result = np.convolve(a_mag, v_mag, mode=mode)
        if a_u is None and v_u is None:
            return result
        if a_u is not None and v_u is not None:
            out_u = str(a_u * v_u)
        else:
            out_u = str(a_u or v_u)
        return _make_result(result, out_u, reg)

    @implements(np.expand_dims)
    def _expand_dims_impl(a: Any, axis: Any) -> Any:
        mag = _mag_of(a)
        unit = _unit_str_of(a)
        reg = _registry_of(a)
        result = np.expand_dims(mag, axis)
        if unit is None:
            return result
        return _make_result(result, unit, reg)

    @implements(np.moveaxis)
    def _moveaxis_impl(a: Any, source: Any, destination: Any) -> Any:
        mag = _mag_of(a)
        unit = _unit_str_of(a)
        reg = _registry_of(a)
        result = np.moveaxis(mag, source, destination)
        if unit is None:
            return result
        return _make_result(result, unit, reg)

    @implements(np.swapaxes)
    def _swapaxes_impl(a: Any, axis1: int, axis2: int) -> Any:
        mag = _mag_of(a)
        unit = _unit_str_of(a)
        reg = _registry_of(a)
        result = np.swapaxes(mag, axis1, axis2)
        if unit is None:
            return result
        return _make_result(result, unit, reg)

    @implements(np.rollaxis)
    def _rollaxis_impl(a: Any, axis: int, start: int = 0) -> Any:
        mag = _mag_of(a)
        unit = _unit_str_of(a)
        reg = _registry_of(a)
        result = np.rollaxis(mag, axis, start)
        if unit is None:
            return result
        return _make_result(result, unit, reg)

    @implements(np.tile)
    def _tile_impl(a: Any, reps: Any) -> Any:
        mag = _mag_of(a)
        unit = _unit_str_of(a)
        reg = _registry_of(a)
        result = np.tile(mag, reps)
        if unit is None:
            return result
        return _make_result(result, unit, reg)

    @implements(np.repeat)
    def _repeat_impl(a: Any, repeats: Any, axis: Any = None) -> Any:
        mag = _mag_of(a)
        unit = _unit_str_of(a)
        reg = _registry_of(a)
        result = np.repeat(mag, repeats, axis=axis)
        if unit is None:
            return result
        return _make_result(result, unit, reg)

    @implements(np.pad)
    def _pad_impl(
        array: Any, pad_width: Any, mode: Any = "constant", **kwargs: Any
    ) -> Any:
        mag = _mag_of(array)
        unit = _unit_str_of(array)
        reg = _registry_of(array)
        if "constant_values" in kwargs:
            cv = kwargs["constant_values"]
            if isinstance(cv, tuple):
                kwargs["constant_values"] = tuple(
                    _convert_mag(_mag_of(c), _unit_str_of(c), unit or "", reg)
                    for c in cv
                )
            else:
                kwargs["constant_values"] = _convert_mag(
                    _mag_of(cv), _unit_str_of(cv), unit or "", reg
                )
        if "end_values" in kwargs:
            ev = kwargs["end_values"]
            if isinstance(ev, tuple):
                kwargs["end_values"] = tuple(
                    _convert_mag(_mag_of(e), _unit_str_of(e), unit or "", reg)
                    for e in ev
                )
            else:
                kwargs["end_values"] = _convert_mag(
                    _mag_of(ev), _unit_str_of(ev), unit or "", reg
                )
        result = np.pad(mag, pad_width, mode=mode, **kwargs)
        if unit is None:
            return result
        return _make_result(result, unit, reg)

    @implements(np.diag)
    def _diag_impl(v: Any, k: int = 0) -> Any:
        mag = _mag_of(v)
        unit = _unit_str_of(v)
        reg = _registry_of(v)
        result = np.diag(mag, k=k)
        if unit is None:
            return result
        return _make_result(result, unit, reg)

    @implements(np.diagonal)
    def _diagonal_impl(a: Any, offset: int = 0, axis1: int = 0, axis2: int = 1) -> Any:
        mag = _mag_of(a)
        unit = _unit_str_of(a)
        reg = _registry_of(a)
        result = np.diagonal(mag, offset=offset, axis1=axis1, axis2=axis2)
        if unit is None:
            return result
        return _make_result(result, unit, reg)

    @implements(np.trace)
    def _trace_impl(  # noqa: PLR0913
        a: Any,
        offset: int = 0,
        axis1: int = 0,
        axis2: int = 1,
        dtype: Any = None,
        out: Any = None,
    ) -> Any:
        mag = _mag_of(a)
        unit = _unit_str_of(a)
        reg = _registry_of(a)
        result = np.trace(
            mag, offset=offset, axis1=axis1, axis2=axis2, dtype=dtype, out=out
        )
        if unit is None:
            return result
        return _make_result(result, unit, reg)

    @implements(np.take)
    def _take_impl(a: Any, indices: Any, axis: Any = None, **kwargs: Any) -> Any:
        mag = _mag_of(a)
        unit = _unit_str_of(a)
        reg = _registry_of(a)
        result = np.take(mag, indices, axis=axis, **kwargs)
        if unit is None:
            return result
        return _make_result(result, unit, reg)

    @implements(np.compress)
    def _compress_impl(
        condition: Any, a: Any, axis: Any = None, out: Any = None
    ) -> Any:
        mag = _mag_of(a)
        unit = _unit_str_of(a)
        reg = _registry_of(a)
        result = np.compress(condition, mag, axis=axis, out=out)
        if unit is None:
            return result
        return _make_result(result, unit, reg)

    @implements(np.unique)
    def _unique_impl(ar: Any, *args: Any, **kwargs: Any) -> Any:
        mag = _mag_of(ar)
        unit = _unit_str_of(ar)
        reg = _registry_of(ar)
        result = np.unique(mag, *args, **kwargs)
        if unit is None:
            return result
        if isinstance(result, tuple):
            head = _make_result(result[0], unit, reg)
            return (head, *result[1:])
        return _make_result(result, unit, reg)

    # hstack / vstack / dstack / column_stack / row_stack
    def _make_stack_impl(np_func: Any) -> Any:
        def _impl(tup: Any, *args: Any, **kwargs: Any) -> Any:
            reg = _first_registry(*tup)
            out_unit: str | None = None
            for a in tup:
                u = _unit_str_of(a)
                if u is not None:
                    out_unit = u
                    break
            mags = [
                _convert_mag(_mag_of(a), _unit_str_of(a), out_unit or "", reg)
                for a in tup
            ]
            result = np_func(mags, *args, **kwargs)
            if out_unit is None:
                return result
            return _make_result(result, out_unit, reg)

        return _impl

    _stack_funcs: tuple[Any, ...] = tuple(
        f
        for f in (
            getattr(np, "hstack", None),
            getattr(np, "vstack", None),
            getattr(np, "dstack", None),
            getattr(np, "column_stack", None),
            getattr(np, "row_stack", None),
        )
        if f is not None
    )
    for _sf in _stack_funcs:
        implements(_sf)(_make_stack_impl(_sf))

    @implements(np.split)
    def _split_impl(ary: Any, indices_or_sections: Any, axis: int = 0) -> Any:
        mag = _mag_of(ary)
        unit = _unit_str_of(ary)
        reg = _registry_of(ary)
        result = np.split(mag, indices_or_sections, axis=axis)
        if unit is None:
            return result
        return [_make_result(r, unit, reg) for r in result]

    @implements(np.array_split)
    def _array_split_impl(ary: Any, indices_or_sections: Any, axis: int = 0) -> Any:
        mag = _mag_of(ary)
        unit = _unit_str_of(ary)
        reg = _registry_of(ary)
        result = np.array_split(mag, indices_or_sections, axis=axis)
        if unit is None:
            return result
        return [_make_result(r, unit, reg) for r in result]

    # nan-aware reductions
    _nan_reducer_funcs: tuple[Any, ...] = tuple(
        f
        for f in (
            getattr(np, "nansum", None),
            getattr(np, "nanmean", None),
            getattr(np, "nanmedian", None),
            getattr(np, "nanmin", None),
            getattr(np, "nanmax", None),
            getattr(np, "nanstd", None),
            getattr(np, "nancumsum", None),
        )
        if f is not None
    )
    for _nf in _nan_reducer_funcs:
        implements(_nf)(_make_unary_reducer(_nf))

    _nanvar = getattr(np, "nanvar", None)
    if _nanvar is not None:
        implements(_nanvar)(_make_unary_reducer(_nanvar, _var_unit_transform))

    @implements(np.nanpercentile)
    def _nanpercentile_impl(a: Any, q: Any, *args: Any, **kwargs: Any) -> Any:
        mag = _mag_of(a)
        unit = _unit_str_of(a)
        reg = _registry_of(a)
        result = np.nanpercentile(mag, q, *args, **kwargs)
        if unit is None:
            return result
        return _make_result(result, unit, reg)

    _nanquantile = getattr(np, "nanquantile", None)
    if _nanquantile is not None:

        def _nanquantile_impl(a: Any, q: Any, *args: Any, **kwargs: Any) -> Any:
            mag = _mag_of(a)
            unit = _unit_str_of(a)
            reg = _registry_of(a)
            result = np.nanquantile(mag, q, *args, **kwargs)
            if unit is None:
                return result
            return _make_result(result, unit, reg)

        implements(_nanquantile)(_nanquantile_impl)

    # np.prod — units change per element; reject dimensioned input like cumprod
    @implements(np.prod)
    def _prod_impl(a: Any, *args: Any, **kwargs: Any) -> Any:
        unit = _unit_str_of(a)
        if unit is not None:
            from pintrs import DimensionalityError  # noqa: PLC0415

            reg = _registry_of(a)
            scalar = reg.Quantity(1.0, unit)
            if not scalar.dimensionless:
                msg = "Cannot apply prod to a dimensioned quantity"
                raise DimensionalityError(msg)
        return np.prod(_mag_of(a), *args, **kwargs)

    @implements(np.result_type)
    def _result_type_impl(*arrays_and_dtypes: Any) -> Any:
        return np.result_type(*(_mag_of(x) for x in arrays_and_dtypes))

    @implements(np.may_share_memory)
    def _may_share_memory_impl(a: Any, b: Any, max_work: Any = None) -> Any:
        return np.may_share_memory(_mag_of(a), _mag_of(b), max_work=max_work)

    @implements(np.shares_memory)
    def _shares_memory_impl(a: Any, b: Any, max_work: Any = None) -> Any:
        return np.shares_memory(_mag_of(a), _mag_of(b), max_work=max_work)

    @implements(np.array_equal)
    def _array_equal_impl(a1: Any, a2: Any, equal_nan: bool = False) -> Any:
        u1 = _unit_str_of(a1)
        u2 = _unit_str_of(a2)
        reg = _first_registry(a1, a2)
        m1 = _mag_of(a1)
        m2 = _mag_of(a2)
        if u1 is not None and u2 is not None and u1 != u2:
            try:
                m2 = _convert_mag(m2, u2, u1, reg)
            except Exception:
                return False
        elif (u1 is None) != (u2 is None):
            return False
        return bool(np.array_equal(m1, m2, equal_nan=equal_nan))

    @implements(np.allclose)
    def _allclose_impl(
        a: Any,
        b: Any,
        rtol: float = 1e-05,
        atol: float = 1e-08,
        equal_nan: bool = False,
    ) -> Any:
        u_a = _unit_str_of(a)
        u_b = _unit_str_of(b)
        reg = _first_registry(a, b)
        m_a = _mag_of(a)
        m_b = _mag_of(b)
        if u_a is not None and u_b is not None:
            m_b = _convert_mag(m_b, u_b, u_a, reg)
        return bool(np.allclose(m_a, m_b, rtol=rtol, atol=atol, equal_nan=equal_nan))

    @implements(np.isclose)
    def _isclose_impl(
        a: Any,
        b: Any,
        rtol: float = 1e-05,
        atol: float = 1e-08,
        equal_nan: bool = False,
    ) -> Any:
        u_a = _unit_str_of(a)
        u_b = _unit_str_of(b)
        reg = _first_registry(a, b)
        m_a = _mag_of(a)
        m_b = _mag_of(b)
        if u_a is not None and u_b is not None:
            m_b = _convert_mag(m_b, u_b, u_a, reg)
        return np.isclose(m_a, m_b, rtol=rtol, atol=atol, equal_nan=equal_nan)

else:

    class _ArrayQuantityUnavailable:
        """Placeholder that fails on use when NumPy is not installed."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            msg = (
                "NumPy support requires numpy. Install pintrs[numpy] or install numpy."
            )
            raise ModuleNotFoundError(msg)

    ArrayQuantity: Any = _ArrayQuantityUnavailable  # type: ignore[no-redef]
