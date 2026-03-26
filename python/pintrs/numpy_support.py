"""NumPy array quantity support for pintrs.

Provides ArrayQuantity that wraps numpy arrays with units, implementing
__array_ufunc__ for transparent numpy integration.
"""

# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportInvalidTypeForm=false, reportPrivateUsage=false, reportRedeclaration=false, reportReturnType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnnecessaryIsInstance=false

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
    from pintrs._core import UnitRegistry as _UnitRegistry

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
        def dimensionality(self) -> str:
            return str(self._unit_obj.dimensionality)

        @property
        def dimensionless(self) -> bool:
            return bool(self._unit_obj.dimensionless)

        @property
        def unitless(self) -> bool:
            return bool(self._unit_obj.dimensionless)

        @property
        def shape(self) -> tuple[int, ...]:
            result: tuple[int, ...] = self._magnitude.shape
            return result

        @property
        def ndim(self) -> int:
            return int(self._magnitude.ndim)

        @property
        def dtype(self) -> np.dtype[Any]:
            return self._magnitude.dtype  # type: ignore[no-any-return]

        @property
        def T(self) -> ArrayQuantity:  # noqa: N802
            return ArrayQuantity(
                self._magnitude.T,
                self._units_str,
                self._registry,
            )

        @property
        def flat(self) -> np.flatiter[Any]:
            return self._magnitude.flat  # type: ignore[no-any-return]

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

        def m_as(self, units: str) -> NDArray[Any]:
            factor = self._registry._get_conversion_factor(
                self._units_str,
                units,
            )
            return self._magnitude * factor

        def to(self, units: str) -> ArrayQuantity:
            factor = self._registry._get_conversion_factor(
                self._units_str,
                units,
            )
            return ArrayQuantity(
                self._magnitude * factor,
                units,
                self._registry,
            )

        def ito(self, units: str) -> None:
            factor = self._registry._get_conversion_factor(
                self._units_str,
                units,
            )
            self._magnitude = self._magnitude * factor
            self._units_str = units
            self._unit_obj = self._registry.Unit(units)

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

        def check(self, dimension: str) -> bool:
            return bool(
                self._registry.Quantity(1.0, self._units_str).check(
                    dimension,
                ),
            )

        def clip(
            self,
            min: float | None = None,  # noqa: A002
            max: float | None = None,  # noqa: A002
        ) -> ArrayQuantity:
            return ArrayQuantity(
                self._magnitude.clip(min, max),
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
            side: str = "left",
        ) -> Any:
            mag = v.magnitude if isinstance(v, (ScalarQuantity, ArrayQuantity)) else v
            return self._magnitude.searchsorted(mag, side=side)

        def dot(
            self,
            other: ArrayQuantity | NDArray[Any],
        ) -> Any:
            mag = other._magnitude if isinstance(other, ArrayQuantity) else other
            return self._magnitude.dot(mag)

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
            if isinstance(value, (ScalarQuantity, ArrayQuantity)):
                self._magnitude[key] = value.magnitude
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
            if not all(issubclass(t, ArrayQuantity) for t in types):
                return NotImplemented

            def _scalar(val: Any, units: str) -> Any:
                return self._registry.Quantity(float(val), units)

            def _array(val: Any, units: str) -> ArrayQuantity:
                return ArrayQuantity(np.asarray(val), units, self._registry)

            if func is np.sum:
                result = np.sum(self._magnitude, **kwargs)
                return _scalar(result, self._units_str)
            if func is np.mean:
                result = np.mean(self._magnitude, **kwargs)
                return _scalar(result, self._units_str)
            if func is np.min or func is np.amin:
                result = np.min(self._magnitude, **kwargs)
                return _scalar(result, self._units_str)
            if func is np.max or func is np.amax:
                result = np.max(self._magnitude, **kwargs)
                return _scalar(result, self._units_str)
            if func is np.std:
                result = np.std(self._magnitude, **kwargs)
                return _scalar(result, self._units_str)
            if func is np.var:
                result = np.var(self._magnitude, **kwargs)
                u = self._unit_obj
                return _scalar(result, str(u * u))
            if func is np.cumsum:
                result = np.cumsum(self._magnitude, **kwargs)
                return _array(result, self._units_str)
            if func is np.cumprod:
                result = np.cumprod(self._magnitude, **kwargs)
                return _array(result, self._units_str)
            if func is np.argmin:
                return np.argmin(self._magnitude, **kwargs)
            if func is np.argmax:
                return np.argmax(self._magnitude, **kwargs)
            if func is np.sort:
                result = np.sort(self._magnitude, **kwargs)
                return _array(result, self._units_str)
            if func is np.argsort:
                return np.argsort(self._magnitude, **kwargs)
            if func is np.isreal:
                return np.isreal(self._magnitude)
            if func is np.iscomplex:
                return np.iscomplex(self._magnitude)
            if func is np.reshape:
                new_shape = (
                    args[1]
                    if len(args) > 1
                    else kwargs.get("newshape", kwargs.get("shape"))
                )
                result = np.reshape(self._magnitude, new_shape)  # pyright: ignore[reportCallIssue]
                return _array(result, self._units_str)
            if func is np.ones_like:
                result = np.ones_like(self._magnitude, **kwargs)
                return _array(result, self._units_str)
            if func is np.zeros_like:
                result = np.zeros_like(self._magnitude, **kwargs)
                return _array(result, self._units_str)
            if func is np.full_like:
                fill = args[1] if len(args) > 1 else kwargs.get("fill_value", 0)
                if isinstance(fill, (ArrayQuantity, ScalarQuantity)):
                    fill = fill.magnitude
                result = np.full_like(self._magnitude, fill, **kwargs)
                return _array(result, self._units_str)
            if func is np.concatenate:
                arrays = args[0]
                mags = [
                    a._magnitude if isinstance(a, ArrayQuantity) else a for a in arrays
                ]
                result = np.concatenate(mags, **kwargs)
                return _array(result, self._units_str)
            if func is np.stack:
                arrays = args[0]
                mags = [
                    a._magnitude if isinstance(a, ArrayQuantity) else a for a in arrays
                ]
                result = np.stack(mags, **kwargs)
                return _array(result, self._units_str)
            if func is np.where:
                condition = args[0]
                if len(args) >= 3:
                    x = (
                        args[1]._magnitude
                        if isinstance(args[1], ArrayQuantity)
                        else args[1]
                    )
                    y = (
                        args[2]._magnitude
                        if isinstance(args[2], ArrayQuantity)
                        else args[2]
                    )
                    result = np.where(condition, x, y)
                    return _array(result, self._units_str)
                return np.where(condition)
            if func is np.clip:
                arr = (
                    args[0]._magnitude
                    if isinstance(args[0], ArrayQuantity)
                    else args[0]
                )
                a_min = kwargs.get("a_min", args[1] if len(args) > 1 else None)
                a_max = kwargs.get("a_max", args[2] if len(args) > 2 else None)
                if isinstance(a_min, (ArrayQuantity, ScalarQuantity)):
                    a_min = a_min.magnitude  # pyright: ignore[reportOptionalMemberAccess]
                if isinstance(a_max, (ArrayQuantity, ScalarQuantity)):
                    a_max = a_max.magnitude  # pyright: ignore[reportOptionalMemberAccess]
                result = np.clip(arr, a_min, a_max)
                return _array(result, self._units_str)
            if func is np.ptp:
                result = np.ptp(self._magnitude, **kwargs)
                return _scalar(result, self._units_str)
            if func is np.median:
                result = np.median(self._magnitude, **kwargs)
                return _scalar(result, self._units_str)
            if func is np.percentile:
                q_val = args[1] if len(args) > 1 else kwargs.get("q")
                result = np.percentile(self._magnitude, q_val, **kwargs)
                return _scalar(result, self._units_str)
            if func is np.dot:
                a, b = args[0], args[1]
                a_mag = a._magnitude if isinstance(a, ArrayQuantity) else np.asarray(a)
                b_mag = b._magnitude if isinstance(b, ArrayQuantity) else np.asarray(b)
                a_u = a._unit_obj if isinstance(a, ArrayQuantity) else None
                b_u = b._unit_obj if isinstance(b, ArrayQuantity) else None
                result = np.dot(a_mag, b_mag, **kwargs)
                if a_u is not None and b_u is not None:
                    out_u = str(a_u * b_u)
                elif a_u is not None:
                    out_u = str(a_u)
                elif b_u is not None:
                    out_u = str(b_u)
                else:
                    out_u = "dimensionless"
                if isinstance(result, np.ndarray):
                    return _array(result, out_u)
                return _scalar(result, out_u)
            if func is np.cross:
                a, b = args[0], args[1]
                a_mag = a._magnitude if isinstance(a, ArrayQuantity) else np.asarray(a)
                b_mag = b._magnitude if isinstance(b, ArrayQuantity) else np.asarray(b)
                a_u = a._unit_obj if isinstance(a, ArrayQuantity) else None
                b_u = b._unit_obj if isinstance(b, ArrayQuantity) else None
                result = np.cross(a_mag, b_mag, **kwargs)
                if a_u is not None and b_u is not None:
                    out_u = str(a_u * b_u)
                elif a_u is not None:
                    out_u = str(a_u)
                elif b_u is not None:
                    out_u = str(b_u)
                else:
                    out_u = "dimensionless"
                if isinstance(result, np.ndarray):
                    return _array(result, out_u)
                return _scalar(result, out_u)
            if func is np.outer:
                a, b = args[0], args[1]
                a_mag = a._magnitude if isinstance(a, ArrayQuantity) else np.asarray(a)
                b_mag = b._magnitude if isinstance(b, ArrayQuantity) else np.asarray(b)
                a_u = a._unit_obj if isinstance(a, ArrayQuantity) else None
                b_u = b._unit_obj if isinstance(b, ArrayQuantity) else None
                result = np.outer(a_mag, b_mag, **kwargs)
                if a_u is not None and b_u is not None:
                    out_u = str(a_u * b_u)
                elif a_u is not None:
                    out_u = str(a_u)
                elif b_u is not None:
                    out_u = str(b_u)
                else:
                    out_u = "dimensionless"
                return _array(result, out_u)
            if func is np.inner:
                a, b = args[0], args[1]
                a_mag = a._magnitude if isinstance(a, ArrayQuantity) else np.asarray(a)
                b_mag = b._magnitude if isinstance(b, ArrayQuantity) else np.asarray(b)
                a_u = a._unit_obj if isinstance(a, ArrayQuantity) else None
                b_u = b._unit_obj if isinstance(b, ArrayQuantity) else None
                result = np.inner(a_mag, b_mag, **kwargs)
                if a_u is not None and b_u is not None:
                    out_u = str(a_u * b_u)
                elif a_u is not None:
                    out_u = str(a_u)
                elif b_u is not None:
                    out_u = str(b_u)
                else:
                    out_u = "dimensionless"
                if isinstance(result, np.ndarray):
                    return _array(result, out_u)
                return _scalar(result, out_u)
            if func is np.vdot:
                a, b = args[0], args[1]
                a_mag = a._magnitude if isinstance(a, ArrayQuantity) else np.asarray(a)
                b_mag = b._magnitude if isinstance(b, ArrayQuantity) else np.asarray(b)
                a_u = a._unit_obj if isinstance(a, ArrayQuantity) else None
                b_u = b._unit_obj if isinstance(b, ArrayQuantity) else None
                result = np.vdot(a_mag, b_mag)
                if a_u is not None and b_u is not None:
                    out_u = str(a_u * b_u)
                elif a_u is not None:
                    out_u = str(a_u)
                elif b_u is not None:
                    out_u = str(b_u)
                else:
                    out_u = "dimensionless"
                return _scalar(result, out_u)
            if func is np.correlate:
                a, b = args[0], args[1]
                a_mag = a._magnitude if isinstance(a, ArrayQuantity) else np.asarray(a)
                b_mag = b._magnitude if isinstance(b, ArrayQuantity) else np.asarray(b)
                a_u = a._unit_obj if isinstance(a, ArrayQuantity) else None
                b_u = b._unit_obj if isinstance(b, ArrayQuantity) else None
                result = np.correlate(a_mag, b_mag, **kwargs)
                if a_u is not None and b_u is not None:
                    out_u = str(a_u * b_u)
                elif a_u is not None:
                    out_u = str(a_u)
                elif b_u is not None:
                    out_u = str(b_u)
                else:
                    out_u = "dimensionless"
                return _array(result, out_u)
            if func is np.trapezoid:
                y = args[0]
                y_mag = y._magnitude if isinstance(y, ArrayQuantity) else np.asarray(y)
                y_u = y._unit_obj if isinstance(y, ArrayQuantity) else None
                trap_kwargs = dict(kwargs)
                x = args[1] if len(args) > 1 else kwargs.get("x")
                if x is not None and "x" in trap_kwargs:
                    del trap_kwargs["x"]
                x_u = None
                if x is not None:
                    if isinstance(x, ArrayQuantity):
                        x_u = x._unit_obj
                        x_mag = x._magnitude
                    else:
                        x_mag = np.asarray(x)
                    result = np.trapezoid(y_mag, x_mag, **trap_kwargs)
                else:
                    result = np.trapezoid(y_mag, **trap_kwargs)
                if y_u is not None and x_u is not None:
                    out_u = str(y_u * x_u)
                elif y_u is not None:
                    out_u = str(y_u)
                elif x_u is not None:
                    out_u = str(x_u)
                else:
                    out_u = "dimensionless"
                if isinstance(result, np.ndarray):
                    return _array(result, out_u)
                return _scalar(result, out_u)
            return NotImplemented

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

else:

    class _ArrayQuantityUnavailable:
        """Placeholder that fails on use when NumPy is not installed."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            msg = (
                "NumPy support requires numpy. Install pintrs[numpy] or install numpy."
            )
            raise ModuleNotFoundError(msg)

    ArrayQuantity: Any = _ArrayQuantityUnavailable  # type: ignore[no-redef]
