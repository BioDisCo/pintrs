"""Logarithmic unit support for pintrs.

Provides LogarithmicQuantity for units like dB, dBm, dBW, Np (neper), and Bel.
These units represent ratios on a logarithmic scale and require special
arithmetic rules.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from pintrs._core import UnitRegistry

_EQ_TOLERANCE = 1e-10


class LogarithmicQuantity:
    """A quantity expressed in logarithmic units (dB, dBm, Np, etc.).

    Logarithmic quantities follow different arithmetic rules than linear
    quantities:
    - Adding two dB values = multiplying the underlying linear values
    - A dBm value has a reference of 1 mW

    Args:
        magnitude: The value in logarithmic units.
        units: The logarithmic unit string ("dB", "dBm", "dBW", "Np", "Bel").
        registry: Optional UnitRegistry for conversions.
    """

    _REFERENCE_POWER: ClassVar[dict[str, tuple[float, str]]] = {
        "dBm": (1e-3, "watt"),
        "dBW": (1.0, "watt"),
        "dBmW": (1e-3, "watt"),
    }

    _LOG_FACTOR: ClassVar[dict[str, float]] = {
        "dB": 10.0,
        "dBm": 10.0,
        "dBW": 10.0,
        "dBmW": 10.0,
        "B": 1.0,
        "Bel": 1.0,
        "Np": 1.0,
    }

    @staticmethod
    def _canonical_units(units: str) -> str:
        return "Bel" if units == "B" else units

    def __init__(
        self,
        magnitude: float,
        units: str = "dB",
        registry: UnitRegistry | None = None,
    ) -> None:
        units = self._canonical_units(units)
        if units not in self._LOG_FACTOR:
            valid = ", ".join(self._LOG_FACTOR)
            msg = f"Unknown logarithmic unit: {units!r}. Valid: {valid}"
            raise ValueError(msg)
        self._magnitude = float(magnitude)
        self._units = units
        self._registry = registry

    @property
    def magnitude(self) -> float:
        """The logarithmic magnitude."""
        return self._magnitude

    @property
    def m(self) -> float:
        """Alias for magnitude."""
        return self._magnitude

    @property
    def units(self) -> str:
        """The logarithmic unit string."""
        return self._units

    def to_linear(self) -> float:
        """Convert to linear scale (ratio).

        For dBm/dBW this returns power in the reference unit (mW or W).
        For dB/Np this returns the dimensionless ratio.
        """
        if self._units == "Np":
            return float(math.exp(2.0 * self._magnitude))
        log_factor = self._LOG_FACTOR[self._units]
        return float(10.0 ** (self._magnitude / log_factor))

    @classmethod
    def from_linear(
        cls,
        value: float,
        units: str = "dB",
        registry: UnitRegistry | None = None,
    ) -> LogarithmicQuantity:
        """Create from a linear value.

        Args:
            value: Linear value (ratio or power).
            units: Target logarithmic unit.
            registry: Optional UnitRegistry.
        """
        units = cls._canonical_units(units)
        if value <= 0:
            msg = "Cannot convert non-positive value to logarithmic scale"
            raise ValueError(msg)
        if units == "Np":
            magnitude = 0.5 * math.log(value)
        else:
            log_factor = cls._LOG_FACTOR[units]
            magnitude = log_factor * math.log10(value)
        return cls(magnitude, units, registry)

    def to(self, units: str) -> LogarithmicQuantity:
        """Convert between logarithmic units.

        Args:
            units: Target logarithmic unit string.
        """
        units = self._canonical_units(units)
        if units == self._units:
            return LogarithmicQuantity(self._magnitude, units, self._registry)

        if units not in self._LOG_FACTOR:
            msg = f"Unknown logarithmic unit: {units!r}"
            raise ValueError(msg)

        src_ref = self._REFERENCE_POWER.get(self._units)
        dst_ref = self._REFERENCE_POWER.get(units)

        if src_ref is not None and dst_ref is not None:
            # Both have absolute references: convert through watts
            src_ref_val = src_ref[0]
            dst_ref_val = dst_ref[0]
            dst_log_factor = self._LOG_FACTOR[units]
            offset = dst_log_factor * math.log10(src_ref_val / dst_ref_val)
            return LogarithmicQuantity(
                self._magnitude + offset,
                units,
                self._registry,
            )

        # Relative or mixed: convert via linear ratio
        linear = self.to_linear()
        return LogarithmicQuantity.from_linear(linear, units, self._registry)

    def to_quantity(self) -> Any:
        """Convert to a linear Quantity (requires a reference unit like dBm/dBW).

        Returns:
            A pintrs Quantity with the linear value and reference unit.

        Raises:
            ValueError: If the unit has no absolute reference (e.g. plain dB).
        """
        ref = self._REFERENCE_POWER.get(self._units)
        if ref is None:
            msg = f"Cannot convert relative unit {self._units!r} to absolute Quantity"
            raise ValueError(msg)
        ref_value, ref_unit = ref
        linear = self.to_linear() * ref_value
        if self._registry is not None:
            return self._registry.Quantity(linear, ref_unit)
        from pintrs._core import Quantity  # noqa: PLC0415

        return Quantity(linear, ref_unit)

    def __repr__(self) -> str:
        return f"<LogarithmicQuantity({self._magnitude}, '{self._units}')>"

    def __str__(self) -> str:
        return f"{self._magnitude} {self._units}"

    def __add__(self, other: LogarithmicQuantity | float) -> LogarithmicQuantity:
        if isinstance(other, LogarithmicQuantity):
            if other._units != self._units:
                other = other.to(self._units)
            return LogarithmicQuantity(
                self._magnitude + other._magnitude,
                self._units,
                self._registry,
            )
        return LogarithmicQuantity(
            self._magnitude + float(other),
            self._units,
            self._registry,
        )

    def __radd__(self, other: float) -> LogarithmicQuantity:
        return self.__add__(other)

    def __sub__(self, other: LogarithmicQuantity | float) -> LogarithmicQuantity:
        if isinstance(other, LogarithmicQuantity):
            if other._units != self._units:
                other = other.to(self._units)
            return LogarithmicQuantity(
                self._magnitude - other._magnitude,
                self._units,
                self._registry,
            )
        return LogarithmicQuantity(
            self._magnitude - float(other),
            self._units,
            self._registry,
        )

    def __rsub__(self, other: float) -> LogarithmicQuantity:
        return LogarithmicQuantity(
            float(other) - self._magnitude,
            self._units,
            self._registry,
        )

    def __neg__(self) -> LogarithmicQuantity:
        return LogarithmicQuantity(-self._magnitude, self._units, self._registry)

    def __mul__(self, other: float) -> LogarithmicQuantity:
        return LogarithmicQuantity(
            self._magnitude * float(other),
            self._units,
            self._registry,
        )

    def __rmul__(self, other: float) -> LogarithmicQuantity:
        return self.__mul__(other)

    def __truediv__(self, other: float) -> LogarithmicQuantity:
        return LogarithmicQuantity(
            self._magnitude / float(other),
            self._units,
            self._registry,
        )

    def __eq__(self, other: object) -> bool:
        if isinstance(other, LogarithmicQuantity):
            if other._units != self._units:
                other = other.to(self._units)
            return abs(self._magnitude - other._magnitude) < _EQ_TOLERANCE
        return NotImplemented

    def __lt__(self, other: LogarithmicQuantity) -> bool:
        if other._units != self._units:
            other = other.to(self._units)
        return self._magnitude < other._magnitude

    def __le__(self, other: LogarithmicQuantity) -> bool:
        if other._units != self._units:
            other = other.to(self._units)
        return self._magnitude <= other._magnitude

    def __gt__(self, other: LogarithmicQuantity) -> bool:
        if other._units != self._units:
            other = other.to(self._units)
        return self._magnitude > other._magnitude

    def __ge__(self, other: LogarithmicQuantity) -> bool:
        if other._units != self._units:
            other = other.to(self._units)
        return self._magnitude >= other._magnitude

    def __hash__(self) -> int:
        return hash((round(self._magnitude, 10), self._units))
