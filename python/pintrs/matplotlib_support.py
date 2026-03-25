"""Matplotlib integration for pintrs.

Registers a unit converter so Quantity values can be plotted directly
with automatic axis labels.
"""

# pyright: basic

from __future__ import annotations

from typing import Any

try:
    import matplotlib.units as munits  # type: ignore[import-not-found,unused-ignore]

    has_matplotlib = True
except ImportError:
    has_matplotlib = False


if has_matplotlib:

    class PintConverter(munits.ConversionInterface):  # type: ignore[misc,unused-ignore]
        """Matplotlib converter for pintrs Quantity objects."""

        @staticmethod
        def convert(value: Any, unit: Any, axis: Any) -> Any:  # noqa: ARG004
            """Convert Quantity to float for plotting."""
            if hasattr(value, "magnitude"):
                if unit is not None and hasattr(value, "to"):
                    return value.to(str(unit)).magnitude
                return value.magnitude
            return value

        @staticmethod
        def axisinfo(unit: Any, axis: Any) -> munits.AxisInfo | None:  # noqa: ARG004
            """Return axis label from unit."""
            if unit is not None:
                return munits.AxisInfo(label=str(unit))  # type: ignore[no-untyped-call]
            return None

        @staticmethod
        def default_units(x: Any, axis: Any) -> Any:  # noqa: ARG004
            """Return the default units for a Quantity."""
            if hasattr(x, "units"):
                return x.units
            if hasattr(x, "__iter__"):
                for item in x:
                    if hasattr(item, "units"):
                        return item.units
            return None


def setup_matplotlib(enable: bool = True) -> None:
    """Enable or disable matplotlib integration for pintrs Quantities.

    Args:
        enable: If True, register the converter. If False, deregister.
    """
    if not has_matplotlib:
        return

    from pintrs._core import Quantity as _Quantity  # noqa: PLC0415

    if enable:
        munits.registry[_Quantity] = PintConverter()  # type: ignore[index,assignment,unused-ignore]
    elif _Quantity in munits.registry:  # type: ignore[operator,unused-ignore]
        del munits.registry[_Quantity]  # type: ignore[attr-defined,unused-ignore]
