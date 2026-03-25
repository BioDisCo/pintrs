"""Parse @group and @system blocks from pint definition files.

The Rust parser handles unit/prefix/dimension definitions but skips
@group and @system blocks. This module parses those blocks and
populates Group and System registries.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pintrs._core import UnitRegistry as _UnitRegistry

from pintrs.group import Group
from pintrs.system import System

_DEFINITIONS_FILE = Path(__file__).parent.parent.parent / "src" / "default_en.txt"

# Fallback: embedded group/system definitions if source file not available
_BUILTIN_GROUPS: dict[str, tuple[list[str], list[str]]] = {}
_BUILTIN_SYSTEMS: dict[str, tuple[list[str], list[str]]] = {}
_parsed = False


def _extract_unit_name(line: str) -> str | None:
    """Extract the primary unit name from a definition line.

    Lines look like: 'inch = yard / 36 = in = international_inch'
    The first token before '=' is the canonical name.
    """
    line = line.strip()
    if not line or line.startswith("#") or line.startswith("@"):
        return None
    # For system rules like 'bohr: meter', extract the unit name
    if ":" in line:
        name = line.split(":")[0].strip()
        return name if name else None
    # For group unit definitions like 'inch = yard / 36 = in'
    name = line.split("=")[0].strip()
    # The name might be a definition, take just the first token
    return name.split()[0] if name else None


def _parse_definition_blocks(text: str) -> None:
    """Parse @group and @system blocks from definition file text."""
    global _parsed  # noqa: PLW0603

    lines = text.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("@group"):
            header = line[len("@group") :].strip()
            # Parse: GroupName [using Group1, Group2, ...]
            using_match = re.match(r"(\S+)\s+using\s+(.+)", header)
            if using_match:
                name = using_match.group(1)
                used = [g.strip() for g in using_match.group(2).split(",")]
            else:
                name = header.split()[0] if header else ""
                used = []

            unit_names: list[str] = []
            i += 1
            while i < len(lines) and lines[i].strip() != "@end":
                uname = _extract_unit_name(lines[i])
                if uname:
                    unit_names.append(uname)
                i += 1
            _BUILTIN_GROUPS[name] = (unit_names, used)

        elif line.startswith("@system"):
            header = line[len("@system") :].strip()
            using_match = re.match(r"(\S+)\s+using\s+(.+)", header)
            if using_match:
                name = using_match.group(1)
                used_groups = [g.strip() for g in using_match.group(2).split(",")]
            else:
                name = header.split()[0] if header else ""
                used_groups = []

            rules: list[str] = []
            i += 1
            while i < len(lines) and lines[i].strip() != "@end":
                stripped = lines[i].strip()
                if stripped and not stripped.startswith("#"):
                    rules.append(stripped)
                i += 1
            _BUILTIN_SYSTEMS[name] = (rules, used_groups)

        i += 1

    _parsed = True


def _load_definitions() -> None:
    """Load and parse the definition file."""
    if _parsed:
        return
    if _DEFINITIONS_FILE.exists():
        text = _DEFINITIONS_FILE.read_text()
        _parse_definition_blocks(text)
    else:
        _use_embedded_definitions()


def _use_embedded_definitions() -> None:
    """Use hardcoded definitions when the source file is not available."""
    global _parsed  # noqa: PLW0603
    # Minimal fallback - the hardcoded groups in group.py handle this case
    _parsed = True


def build_groups_from_definitions(ureg: _UnitRegistry) -> None:
    """Build Group objects from parsed definition file blocks.

    Args:
        ureg: Registry to validate unit names against.
    """
    _load_definitions()

    if not _BUILTIN_GROUPS:
        return

    # First create all groups
    all_registry_units = frozenset(ureg)

    for name, (unit_names, used_groups) in _BUILTIN_GROUPS.items():
        if Group.get(name) is not None:
            continue
        grp = Group(name)
        for uname in unit_names:
            # Add the unit and common aliases/prefixed forms
            if uname in all_registry_units or uname in ureg:
                grp.add_units(uname)
        for used in used_groups:
            grp.add_used_groups(used)

    # Build root group with all units
    if Group.get("root") is None:
        root = Group("root")
        for unit_name in ureg:
            root.add_units(unit_name)


def build_systems_from_definitions(ureg: _UnitRegistry) -> None:
    """Build System objects from parsed definition file blocks.

    Args:
        ureg: Registry for dimensionality lookups.
    """
    _load_definitions()

    if not _BUILTIN_SYSTEMS:
        return

    for name, (rules, used_groups) in _BUILTIN_SYSTEMS.items():
        if System.get(name) is not None:
            continue
        sys = System(name)
        for rule in rules:
            if ":" in rule:
                # Rule format: 'new_unit: old_unit'
                new_unit, old_unit = rule.split(":", 1)
                new_unit = new_unit.strip()
                old_unit = old_unit.strip()
                # Map the dimension of old_unit to new_unit
                try:
                    dim = ureg.get_dimensionality(old_unit)
                    sys.add_rule(dim, new_unit)
                except Exception:  # noqa: S110
                    pass
            else:
                # Simple rule: just a unit name, maps its dimension to itself
                unit_name = rule.strip()
                try:
                    dim = ureg.get_dimensionality(unit_name)
                    sys.add_rule(dim, unit_name)
                except Exception:  # noqa: S110
                    pass

        # Store used groups for reference
        for _gname in used_groups:
            pass  # Group membership is separate from system rules
