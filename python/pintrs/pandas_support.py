"""Pandas ExtensionArray integration for pintrs.

Provides PintDtype and PintArray so that pandas Series and DataFrames
can hold unit-aware columns, similar to pint-pandas.
"""

# pyright: basic, reportAssignmentType=false, reportAttributeAccessIssue=false, reportInvalidTypeForm=false, reportOptionalMemberAccess=false, reportReturnType=false

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import builtins
    from collections.abc import Sequence

    import numpy as np
    from numpy.typing import DTypeLike, NDArray

    from pintrs._core import UnitRegistry as _UnitRegistry

try:
    import numpy as np
    import pandas as pd  # type: ignore[import-untyped]  # noqa: F401
    from pandas.api.extensions import (  # type: ignore[import-untyped]
        ExtensionArray,
        ExtensionDtype,
        register_extension_dtype,
    )

    has_pandas = True
except ImportError:
    has_pandas = False

if has_pandas:
    from pintrs._core import UnitRegistry as _UnitRegistry

    class PintDtype(ExtensionDtype):  # type: ignore[misc]
        """A pandas ExtensionDtype for unit-aware data.

        Args:
            units: Unit string (e.g. "meter", "kg/s").
            registry: Optional UnitRegistry. A default is created if omitted.
        """

        type = float
        na_value = float("nan")

        def __init__(
            self,
            units: str = "dimensionless",
            registry: _UnitRegistry | None = None,
        ) -> None:
            self._units = units
            self._registry = registry or _UnitRegistry()

        @property
        def name(self) -> str:
            return f"pint[{self._units}]"

        @property
        def units(self) -> str:
            return self._units

        @property
        def registry(self) -> _UnitRegistry:
            return self._registry

        @classmethod
        def construct_array_type(cls) -> builtins.type[PintArray]:
            """Return the array type associated with this dtype."""
            return PintArray

        @classmethod
        def construct_from_string(cls, string: str) -> PintDtype:
            """Create a PintDtype from a string like 'pint[meter]'."""
            if not isinstance(string, str):
                msg = f"Expected string, got {type(string)}"
                raise TypeError(msg)
            if string.startswith("pint[") and string.endswith("]"):
                units = string[5:-1]
                return PintDtype(units)
            msg = f"Cannot construct PintDtype from '{string}'"
            raise TypeError(msg)

        def __repr__(self) -> str:
            return f"PintDtype('{self._units}')"

        def __str__(self) -> str:
            return self.name

        def __hash__(self) -> int:
            return hash(self.name)

        def __eq__(self, other: object) -> bool:
            if isinstance(other, PintDtype):
                return self._units == other._units
            if isinstance(other, str):
                return self.name == other
            return NotImplemented

    class PintArray(ExtensionArray):  # type: ignore[misc]
        """A pandas ExtensionArray that holds data with physical units.

        Args:
            values: Numeric array data.
            dtype: PintDtype specifying units.
            copy: Whether to copy the data.
        """

        def __init__(
            self,
            values: NDArray[Any] | Sequence[float],
            dtype: PintDtype | None = None,
            copy: bool = False,
        ) -> None:
            if dtype is None:
                dtype = PintDtype()
            self._dtype = dtype
            arr = np.asarray(values, dtype=float)
            self._data: NDArray[Any] = arr.copy() if copy else arr

        @classmethod
        def _from_sequence(
            cls,
            scalars: Sequence[Any],
            *,
            dtype: PintDtype | None = None,
            copy: bool = False,
        ) -> PintArray:
            values = []
            inferred_units: str | None = None
            for s in scalars:
                if hasattr(s, "magnitude"):
                    values.append(s.magnitude)
                    if inferred_units is None and hasattr(s, "units"):
                        inferred_units = str(s.units)
                elif s is None or (isinstance(s, float) and np.isnan(s)):
                    values.append(float("nan"))
                else:
                    values.append(float(s))
            if dtype is None and inferred_units is not None:
                dtype = PintDtype(inferred_units)
            return PintArray(np.array(values, dtype=float), dtype=dtype, copy=copy)

        @classmethod
        def _from_factorized(
            cls,
            values: NDArray[Any],
            original: PintArray,
        ) -> PintArray:
            return PintArray(values, dtype=original._dtype)

        @property
        def dtype(self) -> PintDtype:
            return self._dtype

        @property
        def units(self) -> str:
            return self._dtype.units

        @property
        def registry(self) -> _UnitRegistry:
            return self._dtype.registry

        def __len__(self) -> int:
            return len(self._data)

        def __getitem__(self, item: Any) -> Any:
            result = self._data[item]
            if isinstance(result, np.ndarray):
                return PintArray(result, dtype=self._dtype)
            if np.isnan(result):
                return self.dtype.na_value
            return self._dtype.registry.Quantity(float(result), self._dtype.units)

        def __setitem__(self, key: Any, value: Any) -> None:
            if hasattr(value, "magnitude"):
                self._data[key] = value.magnitude
            elif value is None:
                self._data[key] = float("nan")
            else:
                self._data[key] = value

        def __repr__(self) -> str:
            return f"PintArray({self._data!r}, units='{self._dtype.units}')"

        def isna(self) -> NDArray[np.bool_]:
            result: NDArray[np.bool_] = np.isnan(self._data)
            return result

        def take(
            self,
            indices: Sequence[int],
            *,
            allow_fill: bool = False,
            fill_value: Any = None,
        ) -> PintArray:
            if allow_fill:
                fill = float("nan") if fill_value is None else float(fill_value)
                data = np.array(
                    [self._data[i] if i >= 0 else fill for i in indices],
                    dtype=float,
                )
            else:
                data = self._data[np.asarray(indices)]
            return PintArray(data, dtype=self._dtype)

        def copy(self) -> PintArray:
            return PintArray(self._data.copy(), dtype=self._dtype)

        @classmethod
        def _concat_same_type(cls, to_concat: Sequence[PintArray]) -> PintArray:
            data = np.concatenate([a._data for a in to_concat])
            return PintArray(data, dtype=to_concat[0]._dtype)

        def _values_for_factorize(self) -> tuple[NDArray[Any], float]:
            return self._data, float("nan")

        def _reduce(self, name: str, *, skipna: bool = True, **kwargs: Any) -> Any:
            funcs: dict[str, Any] = {
                "sum": np.nansum if skipna else np.sum,
                "mean": np.nanmean if skipna else np.mean,
                "median": np.nanmedian if skipna else np.median,
                "min": np.nanmin if skipna else np.min,
                "max": np.nanmax if skipna else np.max,
                "std": np.nanstd if skipna else np.std,
                "var": np.nanvar if skipna else np.var,
            }
            func = funcs.get(name)
            if func is None:
                msg = f"Reduction '{name}' is not supported"
                raise TypeError(msg)
            # numpy reduction functions don't accept pandas-specific kwargs
            kwargs.pop("min_count", None)
            result = func(self._data, **kwargs)
            return self._dtype.registry.Quantity(float(result), self._dtype.units)

        def __array__(self, dtype: DTypeLike | None = None) -> NDArray[Any]:
            if dtype is not None:
                return np.asarray(self._data, dtype=dtype)
            return self._data

        def to(self, units: str) -> PintArray:
            """Convert the array to different units.

            Args:
                units: Target unit string.
            """
            reg = self._dtype.registry
            factor = reg._get_conversion_factor(self._dtype.units, units)
            return PintArray(
                self._data * factor,
                dtype=PintDtype(units, reg),
            )

    register_extension_dtype(PintDtype)

else:

    class _PintDtypeUnavailable:
        """Placeholder that fails on use when pandas is not installed."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            msg = (
                "Pandas support requires pandas and numpy. "
                "Install pintrs[pandas] or install pandas and numpy."
            )
            raise ModuleNotFoundError(msg)

    class _PintArrayUnavailable:
        """Placeholder that fails on use when pandas is not installed."""

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            msg = (
                "Pandas support requires pandas and numpy. "
                "Install pintrs[pandas] or install pandas and numpy."
            )
            raise ModuleNotFoundError(msg)

    PintDtype: Any = _PintDtypeUnavailable  # type: ignore[no-redef]
    PintArray: Any = _PintArrayUnavailable  # type: ignore[no-redef]
