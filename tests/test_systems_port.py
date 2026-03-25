"""System tests - ported from pint's test_systems.py."""

from __future__ import annotations

import pytest
from pintrs import UnitRegistry
from pintrs.group import Group
from pintrs.system import System


@pytest.fixture
def ureg() -> UnitRegistry:
    return UnitRegistry()


class TestSystemCreation:
    def test_create_system(self) -> None:
        sys = System("test_sys_create")
        assert sys.name == "test_sys_create"

    def test_system_with_rules(self) -> None:
        sys = System("test_sys_rules")
        sys.add_rule("[length]", "foot")
        sys.add_rule("[mass]", "pound")
        assert sys.get_rule("[length]") == "foot"
        assert sys.get_rule("[mass]") == "pound"

    def test_system_repr(self) -> None:
        sys = System("test_sys_repr")
        r = repr(sys)
        assert "test_sys_repr" in r

    def test_system_registry(self) -> None:
        name = "test_sys_reg_unique"
        sys = System(name)
        assert System.get(name) is sys

    def test_system_get_nonexistent(self) -> None:
        assert System.get("nonexistent_system_xyz") is None


class TestBuiltinSystems:
    def test_mks_system_exists(self, ureg: UnitRegistry) -> None:
        sys = ureg.get_system("mks")
        assert sys.name == "mks"
        assert len(sys.rules) > 0

    def test_si_system_exists(self, ureg: UnitRegistry) -> None:
        sys = ureg.get_system("SI")
        assert sys.name == "SI"

    def test_cgs_system_exists(self, ureg: UnitRegistry) -> None:
        sys = ureg.get_system("cgs")
        assert sys.name == "cgs"

    def test_imperial_system_exists(self, ureg: UnitRegistry) -> None:
        sys = ureg.get_system("imperial")
        assert sys.name == "imperial"

    def test_us_system_exists(self, ureg: UnitRegistry) -> None:
        sys = ureg.get_system("US")
        assert sys.name == "US"

    def test_gaussian_system_exists(self, ureg: UnitRegistry) -> None:
        sys = ureg.get_system("Gaussian")
        assert sys.name == "Gaussian"

    def test_atomic_system_exists(self, ureg: UnitRegistry) -> None:
        sys = ureg.get_system("atomic")
        assert sys.name == "atomic"

    def test_planck_system_exists(self, ureg: UnitRegistry) -> None:
        sys = ureg.get_system("Planck")
        assert sys.name == "Planck"


class TestSystemRules:
    def test_add_rule(self) -> None:
        sys = System("test_add_rule")
        sys.add_rule("[length]", "meter")
        assert sys.get_rule("[length]") == "meter"

    def test_get_nonexistent_rule(self) -> None:
        sys = System("test_no_rule")
        assert sys.get_rule("[nonexistent]") is None

    def test_multiple_rules(self) -> None:
        sys = System("test_multi_rule")
        sys.add_rule("[length]", "foot")
        sys.add_rule("[mass]", "pound")
        sys.add_rule("[time]", "second")
        assert len(sys.rules) == 3

    def test_override_rule(self) -> None:
        sys = System("test_override_rule")
        sys.add_rule("[length]", "meter")
        sys.add_rule("[length]", "foot")
        assert sys.get_rule("[length]") == "foot"


class TestSystemConversion:
    def test_mks_length(self, ureg: UnitRegistry) -> None:
        sys = ureg.get_system("mks")
        length_unit = sys.get_rule("[length]")
        assert length_unit is not None
        assert "meter" in length_unit or "metre" in length_unit

    def test_cgs_length(self, ureg: UnitRegistry) -> None:
        sys = ureg.get_system("cgs")
        length_unit = sys.get_rule("[length]")
        assert length_unit is not None
        assert "centimeter" in length_unit or "cm" in length_unit

    def test_imperial_length(self, ureg: UnitRegistry) -> None:
        sys = ureg.get_system("imperial")
        length_unit = sys.get_rule("[length]")
        assert length_unit is not None
        assert "yard" in length_unit or "foot" in length_unit


class TestSystemLister:
    def test_sys_property(self, ureg: UnitRegistry) -> None:
        lister = ureg.sys
        assert hasattr(lister, "__getattr__")

    def test_sys_access_mks(self, ureg: UnitRegistry) -> None:
        mks = ureg.sys.mks
        assert mks.name == "mks"

    def test_sys_access_cgs(self, ureg: UnitRegistry) -> None:
        cgs = ureg.sys.cgs
        assert cgs.name == "cgs"

    def test_sys_dir(self, ureg: UnitRegistry) -> None:
        systems = dir(ureg.sys)
        assert "mks" in systems
        assert "cgs" in systems

    def test_sys_repr(self, ureg: UnitRegistry) -> None:
        r = repr(ureg.sys)
        assert "mks" in r


class TestGroupsWithSystems:
    def test_root_group_has_units(self, ureg: UnitRegistry) -> None:
        root = ureg.get_group("root")
        assert len(root.members) > 0

    def test_imperial_group_has_units(self, ureg: UnitRegistry) -> None:
        imperial = ureg.get_group("imperial")
        assert "foot" in imperial

    def test_metric_group_has_units(self, ureg: UnitRegistry) -> None:
        metric = ureg.get_group("metric")
        assert "meter" in metric

    def test_cgs_group_has_units(self, ureg: UnitRegistry) -> None:
        cgs = ureg.get_group("cgs")
        assert "gram" in cgs

    def test_group_cycle_detection(self) -> None:
        g1 = Group("cycle_test_a")
        g2 = Group("cycle_test_b")
        g1.add_used_groups("cycle_test_b")
        with pytest.raises(ValueError, match=r"[Cc]ycl"):
            g2.add_used_groups("cycle_test_a")

    def test_system_from_lines(self) -> None:
        sys = System.from_lines(
            "test_from_lines",
            ["meter", "kilogram", "second"],
            [],
        )
        assert sys.name == "test_from_lines"
