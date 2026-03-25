"""NumPy array quantity support for pintrs.

Provides ArrayQuantity that wraps numpy arrays with units, implementing
__array_ufunc__ for transparent numpy integration.
"""

# pyright: reportPrivateUsage=false, reportUnnecessaryIsInstance=false, reportUnknownArgumentType=false

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
            return self._magnitude.dtype

        @property
        def T(self) -> ArrayQuantity:  # noqa: N802
            return ArrayQuantity(
                self._magnitude.T,
                self._units_str,
                self._registry,
            )

        @property
        def flat(self) -> np.flatiter[Any]:
            return self._magnitude.flat

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
                self._magnitude.clip(min, max),  # type: ignore[arg-type]
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
            if isinstance(value, ScalarQuantity | ArrayQuantity):
                self._magnitude.fill(value.magnitude)
            else:
                self._magnitude.fill(value)

        def put(
            self,
            indices: NDArray[np.intp],
            values: float | ScalarQuantity | ArrayQuantity,
        ) -> None:
            if isinstance(values, ScalarQuantity | ArrayQuantity):
                self._magnitude.put(indices, values.magnitude)
            else:
                self._magnitude.put(indices, values)

        def searchsorted(
            self,
            v: float | ScalarQuantity | ArrayQuantity,
            side: str = "left",
        ) -> Any:
            mag = v.magnitude if isinstance(v, ScalarQuantity | ArrayQuantity) else v
            return self._magnitude.searchsorted(mag, side=side)  # type: ignore[call-overload]

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
            if isinstance(value, ScalarQuantity | ArrayQuantity):
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
            if isinstance(other, np.ndarray | list | tuple | int | float | complex):
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

        @staticmethod
        def _ufunc_output_units(  # noqa: PLR0911
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

            result = getattr(ufunc, method)(*processed, **kwargs)

            if ufunc in self._COMPARISON_UFUNCS:
                return result

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
                if isinstance(magnitude, int | float):
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
            if func is np.sum:
                result = np.sum(self._magnitude, **kwargs)
                return self._registry.Quantity(float(result), self._units_str)
            if func is np.mean:
                result = np.mean(self._magnitude, **kwargs)
                return self._registry.Quantity(float(result), self._units_str)
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

else:
    ArrayQuantity = None  # type: ignore[assignment,misc]
