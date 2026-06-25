"""NumPy array quantity support for pintrs.

Provides ArrayQuantity that wraps numpy arrays with units, implementing
__array_ufunc__ for transparent numpy integration.
"""

# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportCallIssue=false, reportInvalidTypeForm=false, reportOptionalCall=false, reportPossiblyUnboundVariable=false, reportPrivateUsage=false, reportRedeclaration=false, reportReturnType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownParameterType=false, reportUnknownVariableType=false, reportUnnecessaryIsInstance=false, reportUnusedFunction=false

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Any, Callable, Literal, TypeVar

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
    from pintrs._core import DimensionalityError, OffsetUnitCalculusError
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

    def _first_unit_of(items: Any) -> str | None:
        """First non-None unit string among `items`, else None.

        The output unit for sequence ops (concatenate/stack/...) is the unit of
        the first dimensioned operand.
        """
        return next(
            (u for u in (_unit_str_of(a) for a in items) if u is not None), None
        )

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
        if isinstance(mag, (list, tuple)):
            mag = np.asarray(mag, dtype=float)
        return mag * factor

    def _bare_src(
        operand_unit: str | None, ref_unit: str | None, registry: Any
    ) -> str | None:
        """Source unit for an operand combined with a reference quantity.

        A bare (unitless) operand is interpreted as a *dimensionless value* when
        the reference quantity is dimensionless, so it is converted to base units
        like pint (issue #5). For a dimensional reference the operand keeps the
        reference's own unit (existing pintrs convenience).
        """
        if operand_unit is not None:
            return operand_unit
        if (
            ref_unit is not None
            and registry is not None
            and bool(registry.Unit(ref_unit).dimensionless)
        ):
            return "dimensionless"
        return ref_unit

    def _align_operand(
        mag: Any, operand_unit: str | None, ref_unit: str | None, registry: Any
    ) -> Any:
        """Convert an operand's magnitude into ``ref_unit`` for an additive,
        comparison or selection operation, matching pint.

        - A unit-bearing operand is converted (raising on incompatible dims).
        - A bare operand is a dimensionless value: converted to base units when
          the reference is dimensionless. Against a *dimensional* reference only
          zero is allowed (pint permits adding/selecting zero); any other bare
          value raises ``DimensionalityError``.
        """
        if operand_unit is not None:
            dst = ref_unit or ""
            if (
                registry is not None
                and dst
                and (
                    registry._is_non_multiplicative(operand_unit)
                    or registry._is_non_multiplicative(dst)
                )
            ):
                # Offset/log units must convert through their full transform (so
                # 32 degF -> 0 degC, not 17.78), matching pint for the array
                # combination/selection handlers (concatenate, stack, where, ...).
                arr = np.asarray(mag, dtype=float)
                out = np.array(
                    [
                        registry.Quantity(float(x), operand_unit).to(dst).magnitude
                        for x in arr.ravel()
                    ]
                ).reshape(arr.shape)
                return out if arr.ndim else out.item()
            return _convert_mag(mag, operand_unit, dst, registry)
        if ref_unit is None or registry is None:
            return mag
        if bool(registry.Unit(ref_unit).dimensionless):
            return _convert_mag(mag, "dimensionless", ref_unit, registry)
        if bool(np.all(np.asarray(mag) == 0)):
            return mag

        msg = (
            "Cannot combine a bare number with a quantity that has dimensions "
            f"({ref_unit})"
        )
        raise DimensionalityError(msg)

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

        def _convert_magnitude(self, target: str) -> NDArray[Any]:
            # Ordinary units convert by a single factor; offset/logarithmic units
            # (degC, dB, ...) need the offset/log transform applied element-wise,
            # matching scalar Quantity.to().
            reg = self._registry
            src = self._units_str
            if reg._is_non_multiplicative(src) or reg._is_non_multiplicative(target):
                mag = np.asarray(self._magnitude)
                if np.iscomplexobj(mag):
                    return self._convert_complex(mag, src, target)
                return np.array(
                    [
                        reg.Quantity(float(x), src).to(target).magnitude
                        for x in mag.ravel()
                    ]
                ).reshape(mag.shape)
            return self._magnitude * reg._get_conversion_factor(src, target)

        def _convert_complex(
            self, mag: NDArray[Any], src: str, target: str
        ) -> NDArray[Any]:
            """Apply an offset/log conversion to a complex magnitude.

            The scalar ``Quantity.to`` path is real-only, so reconstruct the
            conversion's closed form from real samples and apply it to the
            complex array. The form is chosen from unit metadata, not from the
            sampled curvature, so it never misclassifies:

            - neither side logarithmic -> affine ``f(x) = a + b*x`` (offset and
              ordinary units, including ``degC`` -> ``degF`` and ``dBW`` ->
              ``dBm`` which are both affine);
            - source logarithmic, target not -> exponential ``f(x) = c*r**x``
              (e.g. ``dBW`` -> ``W``);
            - target logarithmic, source not -> logarithmic ``f(x) = p + q*ln(x)``
              (e.g. ``W`` -> ``dBW``).

            This matches pint's element-wise complex behaviour for every
            built-in unit. The exponential form reconstructs its ratio from real
            samples, so a contrived log unit with an extreme ``logfactor`` (no
            real unit has one) can differ from pint in the last few digits.
            """
            reg = self._registry

            def sample(x: float) -> float:
                return float(reg.Quantity(x, src).to(target).magnitude)

            src_log = reg._is_log_unit(src)
            target_log = reg._is_log_unit(target)
            if src_log == target_log:
                # Affine: f(x) = f(0) + (f(1) - f(0)) * x.
                c0 = sample(0.0)
                return (sample(1.0) - c0) * mag + c0
            if src_log:
                # Exponential: f(x) = f(0) * (f(1) / f(0)) ** x.
                c0 = sample(0.0)
                return c0 * (sample(1.0) / c0) ** mag
            # Logarithmic: f(x) = f(1) + (f(e) - f(1)) * ln(x).
            f1 = sample(1.0)
            log_mag: NDArray[Any] = np.log(mag)
            return f1 + (sample(float(np.e)) - f1) * log_mag

        def _ensure_multiplicative(self, *operands: str | None) -> None:
            """Raise OffsetUnitCalculusError if self or any operand has a
            non-multiplicative (offset/log) unit. Multiplying, dividing, or
            powering such units is ambiguous; pint and the scalar Quantity
            enforce the same rule. ``None`` operands (bare numbers) are skipped.
            """
            reg = self._registry
            for unit in (self._units_str, *operands):
                if unit is not None and reg._is_non_multiplicative(unit):
                    msg = "Ambiguous operation with offset unit."
                    raise OffsetUnitCalculusError(msg)

        def m_as(self, units: Any) -> NDArray[Any]:
            return self._convert_magnitude(self._coerce_units(units))

        def to(self, units: Any, *contexts: Any) -> ArrayQuantity:
            u = self._coerce_units(units)
            return ArrayQuantity(self._convert_magnitude(u), u, self._registry)

        def ito(self, units: Any) -> None:
            u = self._coerce_units(units)
            self._magnitude = self._convert_magnitude(u)
            self._units_str = u
            self._unit_obj = self._registry.Unit(u)

        def to_base_units(self) -> ArrayQuantity:
            _factor, unit_str = self._registry._get_root_units(self._units_str)
            return ArrayQuantity(
                self._convert_magnitude(unit_str),
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
            # clip mirrors np.clip: strict on a dimensional array with bare bounds.
            min_c: Any = (
                _align_operand(
                    _mag_of(min), _unit_str_of(min), self._units_str, self._registry
                )
                if min is not None
                else None
            )
            max_c: Any = (
                _align_operand(
                    _mag_of(max), _unit_str_of(max), self._units_str, self._registry
                )
                if max is not None
                else None
            )
            return ArrayQuantity(
                self._magnitude.clip(min_c, max_c),
                self._units_str,
                self._registry,
            )

        def sum(self, axis: int | None = None) -> ArrayQuantity | ScalarQuantity:
            result = self._magnitude.sum(axis=axis)
            if np.ndim(result) == 0:
                return self._registry.Quantity(float(result), self._units_str)
            return ArrayQuantity(np.asarray(result), self._units_str, self._registry)

        def mean(self, axis: int | None = None) -> ArrayQuantity | ScalarQuantity:
            result = self._magnitude.mean(axis=axis)
            if np.ndim(result) == 0:
                return self._registry.Quantity(float(result), self._units_str)
            return ArrayQuantity(np.asarray(result), self._units_str, self._registry)

        def reshape(self, *shape: Any) -> ArrayQuantity:
            if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
                shape = tuple(shape[0])
            return ArrayQuantity(
                self._magnitude.reshape(shape),
                self._units_str,
                self._registry,
            )

        def flatten(self, order: Literal["K", "A", "C", "F"] = "C") -> ArrayQuantity:
            return ArrayQuantity(
                self._magnitude.flatten(order=order),
                self._units_str,
                self._registry,
            )

        def ravel(self, order: Literal["K", "A", "C", "F"] = "C") -> ArrayQuantity:
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
            self._magnitude.fill(self._operand_in_self_units(value))

        def _operand_in_self_units(self, other: Any) -> Any:
            # Express a quantity/bare operand in this array's stored units. A bare
            # value is treated as dimensionless (base) when the array is
            # dimensionless (issue #5), else as the array's own unit (a pintrs
            # convenience kept for assignment/query methods).
            mag, units = self._get_other_mag_and_units(other)
            src = _bare_src(units, self._units_str, self._registry)
            return _convert_mag(mag, src, self._units_str, self._registry)

        def put(
            self,
            indices: NDArray[np.intp],
            values: float | ScalarQuantity | ArrayQuantity,
        ) -> None:
            self._magnitude.put(indices, self._operand_in_self_units(values))

        def searchsorted(
            self,
            v: float | ScalarQuantity | ArrayQuantity,
            side: Literal["left", "right"] = "left",
        ) -> Any:
            mag = self._operand_in_self_units(v)
            return self._magnitude.searchsorted(mag, side=side)  # pyright: ignore[reportCallIssue]

        def dot(
            self,
            other: ArrayQuantity | NDArray[Any],
        ) -> Any:
            mag, units = self._get_other_mag_and_units(other)  # type: ignore[arg-type]
            new_units = self._combine_units(units, operator.mul)
            return _make_result(self._magnitude.dot(mag), new_units, self._registry)

        def copy(self) -> ArrayQuantity:
            """Return a copy of this quantity."""
            return self._wrap(self._magnitude.copy())

        def __len__(self) -> int:
            return len(self._magnitude)

        def __getitem__(
            self,
            key: int | slice | NDArray[np.intp],
        ) -> ArrayQuantity | ScalarQuantity:
            result = self._magnitude[key]
            if isinstance(result, np.ndarray):
                return self._wrap(result)
            return self._registry.Quantity(float(result), self._units_str)

        def __setitem__(
            self,
            key: int | slice | NDArray[np.intp],
            value: float | ScalarQuantity | ArrayQuantity,
        ) -> None:
            self._magnitude[key] = self._operand_in_self_units(value)

        def __iter__(self) -> Iterator[ArrayQuantity | ScalarQuantity]:
            for val in self._magnitude:
                if isinstance(val, np.ndarray):
                    yield self._wrap(val)
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
            return self._wrap(-self._magnitude)

        def __abs__(self) -> ArrayQuantity:
            return self._wrap(np.abs(self._magnitude))

        def conjugate(self) -> ArrayQuantity:
            return self._wrap(np.conjugate(self._magnitude))

        conj = conjugate

        def _wrap(self, mag: Any, units: str | None = None) -> ArrayQuantity:
            """Build an ArrayQuantity in self's registry, defaulting to its units."""
            return ArrayQuantity(
                mag, self._units_str if units is None else units, self._registry
            )

        def _combine_units(
            self,
            units: str | None,
            op: Callable[[Any, Any], Any],
            *,
            reverse: bool = False,
            bare: str | None = None,
        ) -> str:
            """Combine self's unit with operand `units` via `op` (e.g.
            ``operator.mul``). A bare operand (``units is None``) yields `bare`
            (default self's units); `reverse` puts the operand on the left, for
            reflected operators.
            """
            if units is None:
                return self._units_str if bare is None else bare
            other = self._registry.Unit(units)
            left, right = (
                (other, self._unit_obj) if reverse else (self._unit_obj, other)
            )
            return str(op(left, right))

        def _get_other_mag_and_units(
            self,
            other: ArrayQuantity | ScalarQuantity | float,
        ) -> tuple[NDArray[Any] | float, str | None]:
            if isinstance(other, ArrayQuantity):
                return other._magnitude, other._units_str
            if isinstance(other, ScalarQuantity):
                return other.magnitude, str(other.units)
            if _RustArrayQuantity is not None and isinstance(other, _RustArrayQuantity):
                return np.asarray(other.m), other._units_str
            if isinstance(other, _Unit):
                return 1.0, str(other)
            return other, None

        def _dimensionless_factor(self) -> float:
            """Conversion factor to plain dimensionless units.

            Raises ``DimensionalityError`` if the array is not dimensionless.
            Used by operations that mix in a bare number (add/sub/mod/floordiv/
            compare): pint requires a dimensionless quantity and operates on its
            value in base units rather than the intermediate (scaled) units such
            as ``cm*s/m/ms``.
            """
            if not self.dimensionless:
                msg = (
                    "Cannot combine a bare number with a quantity that has "
                    f"dimensions ({self._units_str})"
                )
                raise DimensionalityError(msg)
            return self._registry._get_conversion_factor(
                self._units_str,
                "dimensionless",
            )

        def _bare_base_magnitude(self) -> NDArray[Any]:
            """Magnitude in base (dimensionless) units, for mixing with a bare number."""
            return self._magnitude * self._dimensionless_factor()

        def _bare_compare_magnitude(self, mag: NDArray[Any] | float) -> NDArray[Any]:
            """LHS magnitude for comparing against a bare number.

            Comparison against zero is allowed for any dimensionality (pint);
            otherwise the quantity must be dimensionless and is taken in base
            units.
            """
            if bool(np.all(np.asarray(mag) == 0)):
                return self._magnitude
            return self._bare_base_magnitude()

        def __add__(
            self,
            other: ArrayQuantity | ScalarQuantity | float,
        ) -> ArrayQuantity:
            mag, units = self._get_other_mag_and_units(other)
            if units is None:
                # Adding a bare number/array: zero is allowed for any quantity
                # (pint keeps its units); otherwise require dimensionless and
                # operate in base units.
                if bool(np.all(np.asarray(mag) == 0)):
                    return ArrayQuantity(
                        self._magnitude + mag,
                        self._units_str,
                        self._registry,
                    )
                return ArrayQuantity(
                    self._bare_base_magnitude() + mag,
                    "dimensionless",
                    self._registry,
                )
            special = self._offset_additive(mag, units, subtract=False)
            if special is not None:
                return special
            return ArrayQuantity(
                self._magnitude + self._align_to_self(mag, units),
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
            if units is None:
                if bool(np.all(np.asarray(mag) == 0)):
                    return ArrayQuantity(
                        self._magnitude - mag,
                        self._units_str,
                        self._registry,
                    )
                return ArrayQuantity(
                    self._bare_base_magnitude() - mag,
                    "dimensionless",
                    self._registry,
                )
            special = self._offset_additive(mag, units, subtract=True)
            if special is not None:
                return special
            return ArrayQuantity(
                self._magnitude - self._align_to_self(mag, units),
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
            self._ensure_multiplicative(units)
            new_units = self._combine_units(units, operator.mul)
            return self._wrap(self._magnitude * mag, new_units)

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
            self._ensure_multiplicative(units)
            new_units = self._combine_units(units, operator.truediv)
            return self._wrap(self._magnitude / mag, new_units)

        def __rtruediv__(
            self,
            other: ArrayQuantity | ScalarQuantity | float,
        ) -> ArrayQuantity:
            mag, units = self._get_other_mag_and_units(other)
            self._ensure_multiplicative(units)
            new_units = self._combine_units(
                units,
                operator.truediv,
                reverse=True,
                bare=str(self._registry.Unit("dimensionless") / self._unit_obj),
            )
            return self._wrap(mag / self._magnitude, new_units)

        def __floordiv__(
            self,
            other: ArrayQuantity | ScalarQuantity | float,
        ) -> ArrayQuantity:
            mag, units = self._get_other_mag_and_units(other)
            if units is None:
                # bare number: dimensionless-only, operate in base units.
                return ArrayQuantity(
                    np.floor_divide(self._bare_base_magnitude(), mag),
                    "dimensionless",
                    self._registry,
                )
            # quantity: requires compatible dimensions; yields a plain number.
            factor = self._registry._get_conversion_factor(units, self._units_str)
            return ArrayQuantity(
                np.floor_divide(self._magnitude, mag * factor),
                "dimensionless",
                self._registry,
            )

        def __rfloordiv__(
            self,
            other: ArrayQuantity | ScalarQuantity | float,
        ) -> ArrayQuantity:
            mag, _units = self._get_other_mag_and_units(other)
            return ArrayQuantity(
                np.floor_divide(mag, self._bare_base_magnitude()),
                "dimensionless",
                self._registry,
            )

        def __mod__(
            self,
            other: ArrayQuantity | ScalarQuantity | float,
        ) -> ArrayQuantity:
            mag, units = self._get_other_mag_and_units(other)
            if units is None:
                # bare number: dimensionless-only; pint keeps the dividend's units
                # but operates on the base-unit value.
                factor = self._dimensionless_factor()
                base_remainder = np.mod(self._magnitude * factor, mag)
                return ArrayQuantity(
                    base_remainder / factor,
                    self._units_str,
                    self._registry,
                )
            factor = self._registry._get_conversion_factor(units, self._units_str)
            return ArrayQuantity(
                np.mod(self._magnitude, mag * factor),
                self._units_str,
                self._registry,
            )

        def __rmod__(
            self,
            other: ArrayQuantity | ScalarQuantity | float,
        ) -> ArrayQuantity:
            mag, _units = self._get_other_mag_and_units(other)
            return ArrayQuantity(
                np.mod(mag, self._bare_base_magnitude()),
                "dimensionless",
                self._registry,
            )

        def __divmod__(
            self,
            other: ArrayQuantity | ScalarQuantity | float,
        ) -> tuple[ArrayQuantity, ArrayQuantity]:
            return (self.__floordiv__(other), self.__mod__(other))

        def __rdivmod__(
            self,
            other: ArrayQuantity | ScalarQuantity | float,
        ) -> tuple[ArrayQuantity, ArrayQuantity]:
            return (self.__rfloordiv__(other), self.__rmod__(other))

        def __pow__(self, exp: Any) -> ArrayQuantity:
            exp_mag, exp_units = self._get_other_mag_and_units(exp)
            if exp_units is not None:
                # a quantity exponent must be dimensionless; use its base value
                exp_mag = exp_mag * self._registry._get_conversion_factor(
                    exp_units, "dimensionless"
                )
            if np.ndim(exp_mag) == 0 and exp_mag not in (0, 1):
                # Powering an offset/log unit (other than **0 or **1) is ambiguous.
                self._ensure_multiplicative()
            if np.ndim(exp_mag) > 0:
                # An array exponent is only well-defined (single output unit) when
                # the base is dimensionless, like pint.
                if not self.dimensionless:
                    msg = "Cannot raise a dimensional quantity to an array power"
                    raise DimensionalityError(msg)
                base = self._magnitude * self._dimensionless_factor()
                return ArrayQuantity(base**exp_mag, "dimensionless", self._registry)
            new_units = str(self._unit_obj**exp_mag)
            return ArrayQuantity(self._magnitude**exp_mag, new_units, self._registry)

        def __rpow__(self, base: Any) -> ArrayQuantity:
            # number ** dimensionless-array -> dimensionless (operate in base).
            base_mag, _ = self._get_other_mag_and_units(base)
            exp = self._bare_base_magnitude()
            return ArrayQuantity(
                np.asarray(base_mag) ** exp, "dimensionless", self._registry
            )

        def __matmul__(
            self,
            other: ArrayQuantity | ScalarQuantity | float,
        ) -> ArrayQuantity:
            mag, units = self._get_other_mag_and_units(other)
            new_units = self._combine_units(units, operator.mul)
            return self._wrap(self._magnitude @ mag, new_units)

        def __rmatmul__(
            self,
            other: ArrayQuantity | ScalarQuantity | float,
        ) -> ArrayQuantity:
            mag, units = self._get_other_mag_and_units(other)
            new_units = self._combine_units(units, operator.mul, reverse=True)
            return self._wrap(mag @ self._magnitude, new_units)

        def __eq__(self, other: object) -> Any:
            # Elementwise comparison like pint/numpy: a bare value is compared in
            # base units for a dimensionless quantity; mismatched dimensions give
            # all-False rather than raising.
            if isinstance(other, ArrayQuantity):
                other_unit: str | None = other._units_str
                other_mag = np.asarray(other._magnitude)
            elif isinstance(other, ScalarQuantity):
                other_unit = str(other.units)
                other_mag = np.asarray(other.magnitude)
            elif _RustArrayQuantity is not None and isinstance(
                other, _RustArrayQuantity
            ):
                other_unit = other._units_str
                other_mag = np.asarray(other.m)
            elif isinstance(other, (np.ndarray, list, tuple, int, float, complex)):
                other_unit = None
                other_mag = np.asarray(other)
            else:
                return NotImplemented
            lhs = np.asarray(self._magnitude)
            if other_unit is None:
                if self.dimensionless:
                    return lhs * self._dimensionless_factor() == other_mag
                # Dimensional vs bare: only an all-zero operand compares (zero is
                # zero in any unit, like pint); a non-zero bare operand is unequal.
                if bool(np.all(other_mag == 0)):
                    return lhs == other_mag
                return np.zeros(
                    np.broadcast_shapes(lhs.shape, other_mag.shape), dtype=bool
                )
            try:
                other_mag = _convert_mag(
                    other_mag, other_unit, self._units_str, self._registry
                )
            except Exception:
                return np.zeros(
                    np.broadcast_shapes(lhs.shape, np.shape(other_mag)), dtype=bool
                )
            return lhs == other_mag

        def __ne__(self, other: object) -> Any:
            result = self.__eq__(other)
            if result is NotImplemented:
                return NotImplemented
            return np.logical_not(result)

        def _align_to_self(
            self, mag: NDArray[Any] | float, units: str
        ) -> NDArray[Any] | float:
            """Scale `mag` (expressed in `units`) into self's units.

            A no-op when `units` already matches; the ratio is scale-invariant so
            offset/log units compare correctly without applying the offset.
            """
            if units != self._units_str:
                mag = mag * self._registry._get_conversion_factor(
                    units, self._units_str
                )
            return mag

        def _offset_additive(
            self,
            mag: NDArray[Any] | float,
            units: str,
            *,
            subtract: bool,
        ) -> ArrayQuantity | None:
            """Offset/log additive calculus when both operands are absolute
            (non-multiplicative) units, mirroring the scalar path.

            ``abs + abs`` is ambiguous and raises; ``abs - abs`` yields a delta
            (the element-wise difference after an offset/log-aware conversion).
            Returns ``None`` when this rule does not apply, so the caller uses the
            ordinary additive path (e.g. ``degC + delta_degC``).
            """
            reg = self._registry
            if not (
                reg._is_non_multiplicative(self._units_str)
                and reg._is_non_multiplicative(units)
            ):
                return None
            if not subtract:
                msg = "Ambiguous operation with offset unit."
                raise OffsetUnitCalculusError(msg)
            other = np.asarray(mag, dtype=float)
            conv = np.array(
                [
                    reg.Quantity(float(x), units).to(self._units_str).magnitude
                    for x in other.ravel()
                ]
            ).reshape(other.shape)
            delta_unit = str(
                (
                    reg.Quantity(0.0, self._units_str)
                    - reg.Quantity(0.0, self._units_str)
                ).units
            )
            return ArrayQuantity(self._magnitude - conv, delta_unit, reg)

        def _order(
            self,
            other: ArrayQuantity | ScalarQuantity | float,
            op: Callable[[Any, Any], Any],
        ) -> Any:
            """Shared body of the ordered comparisons; `op` is the operator."""
            if np.iscomplexobj(self._magnitude):
                msg = "Cannot order complex quantities"
                raise TypeError(msg)
            mag, units = self._get_other_mag_and_units(other)
            if units is None:
                return op(self._bare_compare_magnitude(mag), mag)
            return op(self._magnitude, self._align_to_self(mag, units))

        def __lt__(self, other: ArrayQuantity | ScalarQuantity | float) -> Any:
            return self._order(other, operator.lt)

        def __le__(self, other: ArrayQuantity | ScalarQuantity | float) -> Any:
            return self._order(other, operator.le)

        def __gt__(self, other: ArrayQuantity | ScalarQuantity | float) -> Any:
            return self._order(other, operator.gt)

        def __ge__(self, other: ArrayQuantity | ScalarQuantity | float) -> Any:
            return self._order(other, operator.ge)

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
        # Matching-dims ufuncs where a bare operand must be dimensionless
        _BARE_STRICT_UFUNCS: frozenset[np.ufunc] = frozenset()
        # Angular-conversion ufuncs mapped to their target unit. The operand must
        # be an angle or dimensionless (treated as radians); a dimensional operand
        # raises, matching pint.
        _ANGLE_CONVERT_UFUNCS: dict[np.ufunc, str] = {}  # noqa: RUF012

        def _angle_convert(self, target: str, mag: Any, unit: Unit | None) -> Any:
            if unit is None:
                src = "radian"  # a bare operand is treated as radians
            elif bool(unit.dimensionless):
                # dimensionless (possibly scaled): take the base value as radians
                mag = mag * self._registry._get_conversion_factor(
                    str(unit), "dimensionless"
                )
                src = "radian"
            else:
                src = str(unit)  # an angle; conversion below rejects non-angles
            factor = self._registry._get_conversion_factor(src, target)
            return _make_result(mag * factor, target, self._registry)

        def _ufunc_output_units(
            self,
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
                if u0 is not None:
                    return str(u0)
                # bare numerator / quantity -> inverse unit (e.g. 3 / (2 m)).
                return str(u1**-1) if u1 is not None else default_units
            if ufunc in (np.power, np.float_power):
                exp = processed[1] if len(processed) > 1 else 1
                if isinstance(exp, np.ndarray):
                    exp = float(exp.flat[0])
                u0 = input_units[0]
                # A bare or (scaled-)dimensionless base raised to a dimensionless
                # exponent is itself dimensionless, not the base/exponent unit.
                if u0 is None or self._registry.Quantity(1.0, str(u0)).dimensionless:
                    return "dimensionless"
                return str(u0 ** float(exp))
            if ufunc in (np.heaviside, np.sign):
                return "dimensionless"
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
            """Check dimensionality and convert units for add/subtract/compare.

            Unit-bearing operands are aligned to a common unit. For ufuncs that
            require all operands to share dimensionality, a bare (unitless)
            operand is treated as dimensionless: it is rejected against a
            dimensional quantity and converted into the operand's unit for a
            (scaled) dimensionless one, e.g. ``add(1000 ms/s, 1) -> 2000 ms/s``.
            mod/fmod/remainder keep their unit and treat a bare number as raw,
            and arctan2 is scaled in ``_check_trig_units``, so neither takes part
            in the bare-number handling.
            """

            if ufunc not in self._MATCHING_DIMS_UFUNCS:
                return processed
            units_with_idx = [
                (i, u) for i, u in enumerate(input_units) if u is not None
            ]
            if not units_with_idx:
                return processed
            _, base_unit = units_with_idx[0]
            base_dim = str(base_unit.dimensionality)
            result = list(processed)
            for idx, unit in units_with_idx[1:]:
                if str(unit.dimensionality) != base_dim:
                    msg = f"Cannot convert from '{unit}' to '{base_unit}'"
                    raise DimensionalityError(msg)
                if str(unit) != str(base_unit):
                    inp = inputs[idx]
                    if isinstance(inp, ArrayQuantity):
                        result[idx] = inp.to(str(base_unit))._magnitude
                    elif isinstance(inp, ScalarQuantity):
                        result[idx] = inp.to(str(base_unit)).magnitude

            bare_indices = [i for i, u in enumerate(input_units) if u is None]
            if bare_indices and ufunc in self._BARE_STRICT_UFUNCS:
                if not self._registry.Quantity(1.0, str(base_unit)).dimensionless:
                    msg = (
                        f"Cannot apply '{ufunc.__name__}' to '{base_unit}' "
                        f"mixed with a dimensionless value"
                    )
                    raise DimensionalityError(msg)
                factor = self._registry._get_conversion_factor(
                    "dimensionless", str(base_unit)
                )
                if factor != 1.0:
                    for i in bare_indices:
                        result[i] = processed[i] * factor
            return result

        def _require_dimensionless_operand(
            self,
            ufunc: np.ufunc,
            processed: list[NDArray[Any] | float],
            input_units: list[Unit | None],
            idx: int,
        ) -> list[NDArray[Any] | float]:
            """Require operand ``idx`` to be dimensionless and fold in its scale.

            A quantity can be dimensionally dimensionless yet carry a non-unity
            scale, e.g. ``millisecond / second`` converts to plain dimensionless
            with a factor of ``0.001``. NumPy ufuncs see only the raw magnitude,
            so the factor must be applied before evaluating sin/exp/log/power/etc.,
            otherwise ``sin(pi * (8 ms / 8 s))`` would wrongly use ``pi`` instead
            of ``pi * 0.001``. Bare (unitless) operands are left untouched.
            """

            u = input_units[idx]
            if u is None:
                return processed
            u_str = str(u)
            if not self._registry.Quantity(1.0, u_str).dimensionless:
                msg = (
                    f"Cannot apply '{ufunc.__name__}' to quantity with units '{u_str}'"
                )
                raise DimensionalityError(msg)
            factor = self._registry._get_conversion_factor(u_str, "dimensionless")
            if factor == 1.0:
                return processed
            result = list(processed)
            result[idx] = processed[idx] * factor
            return result

        def _check_trig_units(
            self,
            ufunc: np.ufunc,
            inputs: tuple[Any, ...],
            processed: list[NDArray[Any] | float],
            input_units: list[Unit | None],
        ) -> list[NDArray[Any] | float]:
            """Validate and convert units for trig/exp/log/power ufuncs."""

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
                    return self._require_dimensionless_operand(
                        ufunc, processed, input_units, 0
                    )
            if ufunc in self._INVERSE_TRIG_UFUNCS:
                processed = self._require_dimensionless_operand(
                    ufunc, processed, input_units, 0
                )
            if ufunc in self._DIMENSIONLESS_UFUNCS:
                # exp/log are unary, but logaddexp/logaddexp2 are binary: every
                # operand must be dimensionless and scaled by its own factor.
                for idx in range(len(input_units)):
                    processed = self._require_dimensionless_operand(
                        ufunc, processed, input_units, idx
                    )
            # power/float_power: the exponent must be dimensionless (fold in its
            # scale so both the numeric result and the output unit use the true
            # value). An array exponent is only valid on a dimensionless base,
            # since a single output unit cannot represent differing powers.
            if ufunc in (np.power, np.float_power) and len(input_units) > 1:
                processed = self._require_dimensionless_operand(
                    ufunc, processed, input_units, 1
                )
                base_u = input_units[0]
                if base_u is not None and self._registry._is_non_multiplicative(
                    str(base_u)
                ):
                    # Powering an offset/log base (other than **0 or **1) is
                    # ambiguous, like the scalar `**` operator.
                    exp_arr = np.asarray(processed[1])
                    if exp_arr.size != 1 or float(exp_arr.reshape(-1)[0]) not in (
                        0.0,
                        1.0,
                    ):
                        msg = "Ambiguous operation with offset unit."
                        raise OffsetUnitCalculusError(msg)
                if base_u is not None:
                    if self._registry.Quantity(1.0, str(base_u)).dimensionless:
                        # A (scaled-)dimensionless base contributes its true
                        # value, e.g. (2 ms/s) ** n uses 0.002, not 2.
                        factor = self._registry._get_conversion_factor(
                            str(base_u), "dimensionless"
                        )
                        if factor != 1.0:
                            processed = list(processed)
                            processed[0] = processed[0] * factor
                    elif np.asarray(processed[1]).size > 1:
                        msg = (
                            f"Cannot raise '{base_u}' to an array of powers; "
                            f"array exponents require a dimensionless base"
                        )
                        raise DimensionalityError(msg)
            if ufunc in self._ARCTAN2_UFUNCS:
                # Two matching-unit quantities are already aligned by
                # _check_additive_compat (the ratio is scale-invariant). When one
                # operand is a bare number, the quantity must be dimensionless and
                # scaled so it is comparable to the unitless operand.
                units_with_idx = [
                    (i, u) for i, u in enumerate(input_units) if u is not None
                ]
                if len(units_with_idx) == 1:
                    idx, u = units_with_idx[0]
                    if not self._registry.Quantity(1.0, str(u)).dimensionless:
                        msg = (
                            f"Cannot apply 'arctan2' to '{u}' "
                            f"mixed with a dimensionless value"
                        )
                        raise DimensionalityError(msg)
                    processed = self._require_dimensionless_operand(
                        ufunc, processed, input_units, idx
                    )
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
                elif _RustArrayQuantity is not None and isinstance(
                    inp, _RustArrayQuantity
                ):
                    processed.append(np.asarray(inp.m))
                    input_units.append(self._registry.Unit(inp._units_str))
                else:
                    # A bare list/tuple operand must become an ndarray so later
                    # base-unit scaling (``operand * factor``) works.
                    processed.append(
                        np.asarray(inp) if isinstance(inp, (list, tuple)) else inp
                    )
                    input_units.append(None)

            angle_target = self._ANGLE_CONVERT_UFUNCS.get(ufunc)
            if angle_target is not None and method == "__call__":
                return self._angle_convert(angle_target, processed[0], input_units[0])

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

            if isinstance(result, tuple):
                # Multi-output ufuncs (modf, frexp). modf keeps the unit on both
                # parts; frexp's exponent is an integer (dimensionless).
                wrapped: list[Any] = []
                for i, part in enumerate(result):
                    part_units = (
                        "dimensionless" if ufunc is np.frexp and i == 1 else out_units
                    )
                    if isinstance(part, np.ndarray):
                        wrapped.append(ArrayQuantity(part, part_units, self._registry))
                    elif np.isscalar(part):
                        mag = part.item() if isinstance(part, np.generic) else part
                        if isinstance(mag, (int, float)):
                            wrapped.append(
                                self._registry.Quantity(float(mag), part_units)
                            )
                        else:
                            wrapped.append(part)
                    else:
                        wrapped.append(part)
                return tuple(wrapped)
            if isinstance(result, np.ndarray):
                return ArrayQuantity(result, out_units, self._registry)
            if np.isscalar(result):
                magnitude = result.item() if isinstance(result, np.generic) else result
                if isinstance(magnitude, bool):
                    return result
                if isinstance(magnitude, (int, float)):
                    return self._registry.Quantity(float(magnitude), out_units)
                if isinstance(magnitude, (complex, str)):
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

    # Angular-conversion ufuncs: radians/deg2rad -> radian, degrees/rad2deg ->
    # degree (a dimensionless operand is treated as radians; pint parity).
    ArrayQuantity._ANGLE_CONVERT_UFUNCS = {
        np.radians: "radian",
        np.deg2rad: "radian",
        np.degrees: "degree",
        np.rad2deg: "degree",
    }

    # NB: mod/fmod/remainder are intentionally absent. pint applies them to the
    # raw magnitudes (no unit conversion) and keeps the left operand's unit, even
    # for differently-scaled or incompatible units, so they must not be aligned.
    ArrayQuantity._MATCHING_DIMS_UFUNCS = frozenset(
        {
            np.add,
            np.subtract,
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
            np.fmax,
            np.fmin,
            np.hypot,
            np.arctan2,
        },
    )

    # Subset of _MATCHING_DIMS_UFUNCS that require every operand to share
    # dimensionality, so a bare number must be dimensionless (and is converted
    # into the quantity's unit). Excludes mod/fmod/remainder (which keep the
    # unit and treat a bare number as raw) and arctan2 (scaled separately).
    ArrayQuantity._BARE_STRICT_UFUNCS = frozenset(
        {
            np.add,
            np.subtract,
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
            np.fmax,
            np.fmin,
            np.hypot,
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
        start_mag = _align_operand(_mag_of(start), start_u, out_unit, reg)
        stop_mag = _align_operand(_mag_of(stop), stop_u, out_unit, reg)
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
        # logspace exponents must be dimensionless; a dimensionless-scaled
        # Quantity exponent is taken at its base value (issue #5).
        reg = _first_registry(start, stop)
        start_mag = _convert_mag(
            _mag_of(start), _unit_str_of(start), "dimensionless", reg
        )
        stop_mag = _convert_mag(_mag_of(stop), _unit_str_of(stop), "dimensionless", reg)
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
        start_mag = _align_operand(_mag_of(start), start_u, out_unit, reg)
        stop_mag = _align_operand(_mag_of(stop), stop_u, out_unit, reg)
        result = np.geomspace(start_mag, stop_mag, num, endpoint, dtype, axis)
        return _make_result(result, out_unit, reg)

    @implements(np.full)
    def _full_impl(
        shape: Any,
        fill_value: Any,
        dtype: Any = None,
        order: Literal["C", "F"] = "C",
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
        order: Literal["K", "A", "C", "F"] = "K",
        subok: bool = True,
        shape: Any = None,
    ) -> Any:
        reg = _first_registry(a, fill_value)
        a_u = _unit_str_of(a)
        fill_u = _unit_str_of(fill_value)
        unit = a_u or fill_u
        mag = _mag_of(a)
        fill_mag = _mag_of(fill_value)
        if a_u is not None and reg is not None:
            fill_mag = _convert_mag(fill_mag, _bare_src(fill_u, a_u, reg), a_u, reg)
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
        order: Literal["K", "A", "C", "F"] = "K",
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
        order: Literal["K", "A", "C", "F"] = "K",
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
        order: Literal["K", "A", "C", "F"] = "K",
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
        arr_mag = _align_operand(_mag_of(arr), arr_u, out_unit, reg)
        val_mag = _align_operand(_mag_of(values), val_u, out_unit, reg)
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
        casting: Literal[
            "no", "equiv", "safe", "same_kind", "same_value", "unsafe"
        ] = "same_kind",
    ) -> Any:
        reg = _first_registry(*arrays)
        out_unit = _first_unit_of(arrays)
        mags = [
            _align_operand(_mag_of(a), _unit_str_of(a), out_unit, reg) for a in arrays
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
        out_unit = _first_unit_of(arrays)
        mags = [
            _align_operand(_mag_of(a), _unit_str_of(a), out_unit, reg) for a in arrays
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
        x_mag = _align_operand(_mag_of(x), x_u, out_unit, reg)
        y_mag = _align_operand(_mag_of(y), y_u, out_unit, reg)
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
            _align_operand(_mag_of(a_min), _unit_str_of(a_min), unit, reg)
            if a_min is not None
            else None
        )
        a_max_mag = (
            _align_operand(_mag_of(a_max), _unit_str_of(a_max), unit, reg)
            if a_max is not None
            else None
        )
        result = np.clip(a_mag, a_min_mag, a_max_mag, out=out, **kwargs)
        if unit is None:
            return result
        return _make_result(result, unit, reg)

    @implements(np.around)
    @implements(np.round)
    def _around_impl(a: Any, decimals: int = 0, out: Any = None) -> Any:
        reg = _first_registry(a)
        unit = _unit_str_of(a)
        result = np.around(_mag_of(a), decimals=decimals, out=out)
        if unit is None:
            return result
        return _make_result(result, unit, reg)

    # ``np.fix`` is deprecated in NumPy 2.5; fetch it dynamically so static
    # analysis does not flag the deprecation, register a handler for parity, but
    # compute with the equivalent (non-deprecated) ``np.trunc`` internally.
    _np_fix_fn: Any = getattr(np, "fix", None)
    if _np_fix_fn is not None:

        @implements(_np_fix_fn)
        def _fix_impl(x: Any, out: Any = None) -> Any:
            reg = _first_registry(x)
            unit = _unit_str_of(x)
            result = np.trunc(_mag_of(x), out=out)
            if unit is None:
                return result
            return _make_result(result, unit, reg)

    @implements(np.average)
    def _average_impl(
        a: Any,
        axis: Any = None,
        weights: Any = None,
        returned: bool = False,
    ) -> Any:
        reg = _first_registry(a)
        unit = _unit_str_of(a)
        w = _mag_of(weights) if weights is not None else None
        result = np.average(_mag_of(a), axis=axis, weights=w, returned=returned)  # type: ignore[call-overload]
        if unit is None:
            return result
        if returned:
            avg, sum_w = result
            return (_make_result(avg, unit, reg), sum_w)
        return _make_result(result, unit, reg)

    @implements(np.insert)
    def _insert_impl(arr: Any, obj: Any, values: Any, axis: int | None = None) -> Any:
        reg = _first_registry(arr, values)
        unit = _unit_str_of(arr)
        val_mag = _align_operand(_mag_of(values), _unit_str_of(values), unit, reg)
        result = np.insert(_mag_of(arr), obj, val_mag, axis=axis)
        if unit is None:
            return result
        return _make_result(result, unit, reg)

    @implements(np.nan_to_num)
    def _nan_to_num_impl(
        x: Any,
        copy: bool = True,
        nan: float = 0.0,
        posinf: float | None = None,
        neginf: float | None = None,
    ) -> Any:
        reg = _first_registry(x)
        unit = _unit_str_of(x)

        def _repl(v: Any) -> float | None:
            # Replacement values are aligned to the array's units: a quantity is
            # converted; a bare value is dimensionless (base for a dimensionless
            # array, zero allowed, else DimensionalityError).
            if v is None:
                return None
            v_mag = np.asarray(_mag_of(v), dtype=float)
            return float(np.asarray(_align_operand(v_mag, _unit_str_of(v), unit, reg)))

        result = np.nan_to_num(
            _mag_of(x),
            copy=copy,
            nan=_repl(nan) or 0.0,
            posinf=_repl(posinf),
            neginf=_repl(neginf),
        )
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
        # Shape/axis transforms and percentile-style reductions: each preserves
        # the unit and forwards its args unchanged, so the unary-reducer factory
        # covers them exactly.
        np.reshape,
        np.percentile,
        np.quantile,
        np.expand_dims,
        np.moveaxis,
        np.swapaxes,
        np.rollaxis,
        np.tile,
        np.repeat,
        np.diag,
        np.diagonal,
        np.trace,
        np.take,
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
        mag = _mag_of(a)
        if unit is not None:
            reg = _registry_of(a)
            scalar = reg.Quantity(1.0, unit)
            if not scalar.dimensionless:
                msg = "Cannot apply cumprod to a dimensioned quantity"
                raise DimensionalityError(msg)
            # Operate on base (dimensionless) values, like pint.
            factor = reg._get_conversion_factor(unit, "dimensionless")
            result = np.cumprod(mag * factor, *args, **kwargs)
            return _make_result(result, "dimensionless", reg)
        return np.cumprod(mag, *args, **kwargs)

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
                # dx may itself be a quantity; integrate over its magnitude and
                # carry its unit into the result (pint parity).
                x_u = _unit_obj_of(dx)
                result = _trap_fn(y_mag, dx=_mag_of(dx), axis=axis)
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

    @implements(np.einsum)
    def _einsum_impl(*operands: Any, **kwargs: Any) -> Any:
        # einsum sums products of one element from each operand, so the result
        # unit is the product of all operand units (pint parity). Non-array
        # arguments (the subscripts string, index sublists) pass through.
        reg = _first_registry(*operands)
        out_u: Any = None
        processed: list[Any] = []
        for o in operands:
            u = _unit_obj_of(o)
            if u is not None:
                out_u = u if out_u is None else out_u * u
                processed.append(_mag_of(o))
            else:
                processed.append(o)
        result = np.einsum(*processed, **kwargs)
        if out_u is None:
            return result
        return _make_result(result, str(out_u), reg)

    @implements(np.unwrap)
    def _unwrap_impl(p: Any, *args: Any, **kwargs: Any) -> Any:
        # Unwrap operates on the magnitude (phase) and preserves the unit.
        unit = _unit_str_of(p)
        reg = _registry_of(p)
        result = np.unwrap(_mag_of(p), *args, **kwargs)
        if unit is None:
            return result
        return _make_result(result, unit, reg)

    @implements(np.searchsorted)
    def _searchsorted_free_impl(
        a: Any, v: Any, side: Literal["left", "right"] = "left", sorter: Any = None
    ) -> Any:
        reg = _first_registry(a, v)
        a_unit = _unit_str_of(a)
        v_mag = _align_operand(_mag_of(v), _unit_str_of(v), a_unit, reg)
        return np.searchsorted(_mag_of(a), v_mag, side=side, sorter=sorter)

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

        if isinstance(result, (list, tuple)):
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
        x_mag = _align_operand(_mag_of(x), _unit_str_of(x), xp_unit, reg)
        xp_mag = _mag_of(xp)
        fp_mag = _mag_of(fp)
        fp_unit = _unit_str_of(fp)
        left_mag = (
            _align_operand(_mag_of(left), _unit_str_of(left), fp_unit, reg)
            if left is not None
            else None
        )
        right_mag = (
            _align_operand(_mag_of(right), _unit_str_of(right), fp_unit, reg)
            if right is not None
            else None
        )
        period_mag = (
            _align_operand(_mag_of(period), _unit_str_of(period), xp_unit, reg)
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
    def _convolve_impl(
        a: Any, v: Any, mode: Literal["valid", "same", "full"] = "full"
    ) -> Any:
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

    @implements(np.pad)
    def _pad_impl(
        array: Any, pad_width: Any, mode: Any = "constant", **kwargs: Any
    ) -> Any:
        mag = _mag_of(array)
        unit = _unit_str_of(array)
        reg = _registry_of(array)

        def _pad_val(v: Any) -> Any:
            return _align_operand(_mag_of(v), _unit_str_of(v), unit, reg)

        if "constant_values" in kwargs:
            cv = kwargs["constant_values"]
            if isinstance(cv, tuple):
                kwargs["constant_values"] = tuple(_pad_val(c) for c in cv)
            else:
                kwargs["constant_values"] = _pad_val(cv)
        if "end_values" in kwargs:
            ev = kwargs["end_values"]
            if isinstance(ev, tuple):
                kwargs["end_values"] = tuple(_pad_val(e) for e in ev)
            else:
                kwargs["end_values"] = _pad_val(ev)
        result = np.pad(mag, pad_width, mode=mode, **kwargs)
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
            out_unit = _first_unit_of(tup)
            mags = [
                _align_operand(_mag_of(a), _unit_str_of(a), out_unit, reg) for a in tup
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
    def _prod_impl(a: Any, axis: Any = None, *args: Any, **kwargs: Any) -> Any:
        unit = _unit_str_of(a)
        mag = np.asarray(_mag_of(a))
        result = np.prod(mag, axis, *args, **kwargs)
        if unit is None:
            return result
        reg = _registry_of(a) or _first_registry(a)
        # Each factor multiplies the unit, so the product carries ``unit ** n``
        # where n is the number of elements reduced (pint parity).
        n = int(mag.size if axis is None else mag.shape[axis])
        return _make_result(result, str(reg.Unit(unit) ** n), reg)

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
        ref = u_a or u_b
        if ref is not None:
            m_a = _align_operand(m_a, u_a, ref, reg)
            m_b = _align_operand(m_b, u_b, ref, reg)
        # atol shares the operands' units (a quantity is converted); rtol is
        # dimensionless. Extract plain magnitudes so numpy does not recurse.
        atol = _convert_mag(_mag_of(atol), _unit_str_of(atol), ref or "", reg)
        rtol = _mag_of(rtol)
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
        ref = u_a or u_b
        if ref is not None:
            m_a = _align_operand(m_a, u_a, ref, reg)
            m_b = _align_operand(m_b, u_b, ref, reg)
        atol = _convert_mag(_mag_of(atol), _unit_str_of(atol), ref or "", reg)
        rtol = _mag_of(rtol)
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
