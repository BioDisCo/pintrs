"""Unit system support for pintrs.

Systems define preferred base units for each dimension. For example,
the "mks" system maps [length]->meter, [mass]->kilogram, [time]->second.
"""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from pintrs._core import UnitRegistry as _UnitRegistry


class System:
    """A coherent set of preferred units for each dimension.

    Systems map dimensions to their preferred base units. For example,
    "mks" maps [length] to meter, [mass] to kilogram, and [time] to second.
    This allows converting quantities to a system's preferred representation.

    Args:
        name: Name of the system.
    """

    _REGISTRY: ClassVar[dict[str, System]] = {}

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._rules: dict[str, str] = {}
        if name:
            System._REGISTRY[name] = self

    def add_rule(self, dimension: str, unit: str) -> None:
        """Map a dimension to its preferred unit in this system.

        Args:
            dimension: Dimensionality string (e.g. "[length]").
            unit: Preferred unit name (e.g. "meter").
        """
        self._rules[dimension] = unit

    def remove_rule(self, dimension: str) -> None:
        """Remove a dimension mapping.

        Args:
            dimension: Dimensionality string to remove.
        """
        self._rules.pop(dimension, None)

    @property
    def rules(self) -> dict[str, str]:
        """Mapping of dimensions to preferred units."""
        return dict(self._rules)

    @property
    def base_units(self) -> dict[str, str]:
        """Mapping of dimension -> preferred unit."""
        return dict(self._rules)

    def preferred_unit_for(self, dimension: str) -> str | None:
        """Get the preferred unit for a dimension.

        Args:
            dimension: Dimensionality string.

        Returns:
            Unit name, or None if not defined.
        """
        return self._rules.get(dimension)

    def get_rule(self, dimension: str) -> str | None:
        """Compatibility alias for preferred_unit_for."""
        return self.preferred_unit_for(dimension)

    def convert(
        self,
        ureg: _UnitRegistry,
        value: float,
        src_units: str,
    ) -> tuple[float, str]:
        """Convert a value to this system's preferred units.

        Args:
            ureg: Unit registry for conversions.
            value: Magnitude to convert.
            src_units: Source unit string.

        Returns:
            Tuple of (converted_magnitude, target_unit_string).

        Raises:
            KeyError: If no preferred unit is defined for the dimension.
        """
        dim = ureg.get_dimensionality(src_units)
        target = self._rules.get(str(dim))
        if target is None:
            return value, src_units
        factor = ureg._get_conversion_factor(src_units, target)
        return value * factor, target

    def __repr__(self) -> str:
        return f"<System('{self.name}', {len(self._rules)} rules)>"

    def __contains__(self, dimension: str) -> bool:
        return dimension in self._rules

    @staticmethod
    def get(name: str) -> System | None:
        """Look up a system by name."""
        return System._REGISTRY.get(name)

    @classmethod
    def from_lines(
        cls,
        lines: list[str] | str,
        units: list[str] | None = None,
        used_groups: list[str] | None = None,  # noqa: ARG003
    ) -> System:
        """Create a System from definition lines.

        Args:
            lines: Lines from a system definition block.
        """
        if isinstance(lines, str):
            sys = cls(lines)
            if units:
                from pintrs._core import UnitRegistry  # noqa: PLC0415

                ureg = UnitRegistry()
                for unit in units:
                    dim = ureg.get_dimensionality(unit)
                    sys.add_rule(str(dim), unit)
            return sys

        name = lines[0].strip() if lines else ""
        sys = cls(name)
        for raw_line in lines[1:]:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            dim, _, unit = stripped.partition(":")
            if unit:
                sys.add_rule(dim.strip(), unit.strip())
        return sys


def _build_builtin_systems() -> None:
    """Build the built-in unit systems."""
    if "mks" in System._REGISTRY:
        return

    mks = System("mks")
    mks.add_rule("[length]", "meter")
    mks.add_rule("[mass]", "kilogram")
    mks.add_rule("[time]", "second")
    mks.add_rule("[temperature]", "kelvin")
    mks.add_rule("[current]", "ampere")
    mks.add_rule("[substance]", "mole")
    mks.add_rule("[luminosity]", "candela")

    si = System("SI")
    for dim, unit in mks.base_units.items():
        si.add_rule(dim, unit)

    cgs = System("cgs")
    cgs.add_rule("[length]", "centimeter")
    cgs.add_rule("[mass]", "gram")
    cgs.add_rule("[time]", "second")
    cgs.add_rule("[temperature]", "kelvin")
    cgs.add_rule("[current]", "ampere")

    imperial = System("imperial")
    imperial.add_rule("[length]", "foot")
    imperial.add_rule("[mass]", "pound")
    imperial.add_rule("[time]", "second")
    imperial.add_rule("[temperature]", "degree_Fahrenheit")

    us = System("US")
    us.add_rule("[length]", "foot")
    us.add_rule("[mass]", "pound")
    us.add_rule("[time]", "second")
    us.add_rule("[temperature]", "degree_Fahrenheit")

    gaussian = System("Gaussian")
    gaussian.add_rule("[length]", "centimeter")
    gaussian.add_rule("[mass]", "gram")
    gaussian.add_rule("[time]", "second")

    atomic = System("atomic")
    atomic.add_rule("[length]", "bohr")
    atomic.add_rule("[mass]", "electron_mass")
    atomic.add_rule("[time]", "atomic_unit_of_time")
    atomic.add_rule("[temperature]", "atomic_unit_of_temperature")
    atomic.add_rule("[current]", "atomic_unit_of_current")

    planck = System("Planck")
    planck.add_rule("[length]", "planck_length")
    planck.add_rule("[mass]", "planck_mass")
    planck.add_rule("[time]", "planck_time")
    planck.add_rule("[temperature]", "planck_temperature")


_build_builtin_systems()
