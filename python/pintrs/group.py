"""Unit group support for pintrs.

Groups are named collections of units. For example, the "imperial" group
contains foot, inch, yard, mile, etc. Groups can include other groups
and support adding/removing units.
"""

# pyright: reportPrivateUsage=false, reportUnusedFunction=false

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pintrs._core import UnitRegistry as _UnitRegistry


class Group:
    """A named collection of units.

    Groups provide a way to organize units into logical sets (e.g., "imperial",
    "metric", "US_customary"). A special "root" group contains all units
    known to the registry.

    Args:
        name: Name of the group.
    """

    _REGISTRY: ClassVar[dict[str, Group]] = {}

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._units: set[str] = set()
        self._used_groups: set[str] = set()
        if name:
            Group._REGISTRY[name] = self

    def add_units(self, *units: str) -> None:
        """Add units to this group.

        Args:
            units: Unit name strings to add.
        """
        self._units.update(units)

    def remove_units(self, *units: str) -> None:
        """Remove units from this group.

        Args:
            units: Unit name strings to remove.
        """
        self._units -= set(units)

    def add_used_groups(self, *groups: str) -> None:
        """Include other groups' units in this group.

        Args:
            groups: Group names to include.

        Raises:
            ValueError: If adding a group would create a cycle.
        """
        for gname in groups:
            if self._would_create_cycle(gname):
                msg = f"Adding group '{gname}' to '{self.name}' would create a cycle"
                raise ValueError(msg)
        self._used_groups.update(groups)

    def _would_create_cycle(self, candidate: str) -> bool:
        """Check if adding candidate group would create a circular dependency."""
        if candidate == self.name:
            return True
        visited: set[str] = set()
        return self._has_path_to(candidate, self.name, visited)

    @staticmethod
    def _has_path_to(start: str, target: str, visited: set[str]) -> bool:
        """Check if there's a path from start to target via used_groups."""
        if start in visited:
            return False
        visited.add(start)
        grp = Group._REGISTRY.get(start)
        if grp is None:
            return False
        for used in grp._used_groups:
            if used == target:
                return True
            if Group._has_path_to(used, target, visited):
                return True
        return False

    def remove_used_groups(self, *groups: str) -> None:
        """Stop including other groups' units.

        Args:
            groups: Group names to exclude.
        """
        self._used_groups -= set(groups)

    @property
    def members(self) -> frozenset[str]:
        """All units in this group, including from included groups."""
        result = set(self._units)
        for gname in self._used_groups:
            grp = Group._REGISTRY.get(gname)
            if grp is not None:
                result |= grp.members
        return frozenset(result)

    def __contains__(self, unit: str) -> bool:
        return unit in self.members

    def __iter__(self) -> Iterator[str]:
        return iter(self.members)

    def __len__(self) -> int:
        return len(self.members)

    def __repr__(self) -> str:
        return f"<Group('{self.name}', {len(self.members)} units)>"

    @staticmethod
    def get(name: str) -> Group | None:
        """Look up a group by name."""
        return Group._REGISTRY.get(name)

    @classmethod
    def from_lines(cls, lines: list[str]) -> Group:
        """Create a Group from definition lines.

        Args:
            lines: Lines from a group definition block.
        """
        name = lines[0].strip() if lines else ""
        grp = cls(name)
        for raw_line in lines[1:]:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            for token in stripped.replace(",", " ").split():
                grp.add_units(token)
        return grp


_IMPERIAL_LENGTH = [
    "foot",
    "inch",
    "yard",
    "mile",
    "thou",
    "fathom",
    "furlong",
    "chain",
    "rod",
    "league",
    "hand",
    "link",
    "survey_foot",
    "survey_mile",
    "cables_length",
    "nautical_mile",
]
_IMPERIAL_MASS = [
    "pound",
    "ounce",
    "stone",
    "grain",
    "dram",
    "quarter",
    "hundredweight",
    "long_hundredweight",
    "long_ton",
    "UK_hundredweight",
    "UK_ton",
    "troy_ounce",
    "troy_pound",
    "pennyweight",
    "scruple",
    "apothecary_ounce",
    "apothecary_pound",
    "apothecary_dram",
    "carat",
    "slug",
    "slinch",
]
_IMPERIAL_VOLUME = [
    "imperial_gallon",
    "imperial_quart",
    "imperial_pint",
    "imperial_cup",
    "imperial_fluid_ounce",
    "imperial_gill",
    "imperial_barrel",
    "imperial_bushel",
    "imperial_peck",
]
_IMPERIAL_TEMP = ["degree_Fahrenheit", "degree_Rankine"]
_IMPERIAL_FORCE = [
    "force_pound",
    "force_ounce",
    "UK_force_ton",
    "US_force_ton",
    "poundal",
    "kip",
]
_IMPERIAL_PRESSURE = [
    "psi",
    "inch_Hg",
    "inch_Hg_32F",
    "inch_Hg_60F",
    "inch_H2O_39F",
    "inch_H2O_60F",
    "foot_Hg",
    "foot_Hg_0C",
    "foot_H2O",
    "foot_H2O_4C",
    "foot_H2O_60F",
]
_IMPERIAL_ENERGY = [
    "british_thermal_unit",
    "foot_pound",
]
_IMPERIAL_POWER = [
    "horsepower",
    "mechanical_horsepower",
    "electrical_horsepower",
    "boiler_horsepower",
]

_US_VOLUME = [
    "US_liquid_gallon",
    "US_liquid_quart",
    "US_liquid_pint",
    "US_liquid_cup",
    "US_fluid_ounce",
    "US_liquid_gill",
    "US_dry_gallon",
    "US_dry_quart",
    "US_dry_pint",
    "US_bushel",
    "US_peck",
    "US_dry_barrel",
    "US_liquid_barrel",
    "barrel",
]

_METRIC_LENGTH = ["meter"]
_METRIC_MASS = ["gram", "metric_ton"]
_METRIC_VOLUME = ["liter"]
_METRIC_TEMP = ["degree_Celsius", "kelvin"]
_METRIC_FORCE = ["newton", "dyne", "force_kilogram", "force_gram"]
_METRIC_PRESSURE = ["pascal", "bar", "atmosphere", "torr", "millimeter_Hg"]
_METRIC_ENERGY = ["joule", "calorie", "erg", "electron_volt"]
_METRIC_POWER = ["watt"]
_METRIC_TIME = ["second", "minute", "hour", "day"]


def _ensure_root_group(ureg: _UnitRegistry | None = None) -> Group:
    """Ensure the root group exists and return it."""
    existing = Group.get("root")
    if existing is not None:
        return existing
    root = Group("root")
    if ureg is not None:
        for unit_name in ureg:
            root.add_units(unit_name)
    return root


def _build_builtin_groups(ureg: _UnitRegistry | None = None) -> None:
    """Build built-in groups with well-known unit assignments.

    Args:
        ureg: Optional registry to validate unit names against.
    """
    root = _ensure_root_group(ureg)

    if "imperial" in Group._REGISTRY:
        return

    imperial = Group("imperial")
    all_imperial = (
        _IMPERIAL_LENGTH
        + _IMPERIAL_MASS
        + _IMPERIAL_VOLUME
        + _IMPERIAL_TEMP
        + _IMPERIAL_FORCE
        + _IMPERIAL_PRESSURE
        + _IMPERIAL_ENERGY
        + _IMPERIAL_POWER
    )
    valid_units = root.members if ureg is not None else None
    for u in all_imperial:
        if valid_units is None or u in valid_units:
            imperial.add_units(u)

    us = Group("US_customary")
    us.add_used_groups("imperial")
    for u in _US_VOLUME:
        if valid_units is None or u in valid_units:
            us.add_units(u)

    metric = Group("metric")
    all_metric = (
        _METRIC_LENGTH
        + _METRIC_MASS
        + _METRIC_VOLUME
        + _METRIC_TEMP
        + _METRIC_FORCE
        + _METRIC_PRESSURE
        + _METRIC_ENERGY
        + _METRIC_POWER
        + _METRIC_TIME
    )
    for u in all_metric:
        if valid_units is None or u in valid_units:
            metric.add_units(u)

    cgs = Group("cgs")
    cgs_units = [
        "centimeter",
        "gram",
        "second",
        "dyne",
        "erg",
        "barye",
        "poise",
        "stokes",
        "gauss",
        "oersted",
        "maxwell",
        "stilb",
        "phot",
        "gal",
    ]
    for u in cgs_units:
        if valid_units is None or u in valid_units:
            cgs.add_units(u)
