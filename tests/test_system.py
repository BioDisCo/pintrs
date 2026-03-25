"""Tests for pintrs System functionality."""

from __future__ import annotations

import pintrs
from pintrs.system import System


def test_system_creation():
    sys = System("test_sys_1")
    assert sys.name == "test_sys_1"
    assert sys.base_units == {}


def test_system_add_remove_rule():
    sys = System("test_sys_2")
    sys.add_rule("[length]", "meter")
    sys.add_rule("[mass]", "kilogram")
    assert sys.base_units == {"[length]": "meter", "[mass]": "kilogram"}

    sys.remove_rule("[mass]")
    assert sys.base_units == {"[length]": "meter"}


def test_system_preferred_unit_for():
    sys = System("test_sys_3")
    sys.add_rule("[length]", "meter")
    assert sys.preferred_unit_for("[length]") == "meter"
    assert sys.preferred_unit_for("[mass]") is None


def test_system_contains():
    sys = System("test_sys_4")
    sys.add_rule("[length]", "meter")
    assert "[length]" in sys
    assert "[mass]" not in sys


def test_system_registry_lookup():
    sys = System("test_sys_lookup")
    assert System.get("test_sys_lookup") is sys
    assert System.get("nonexistent_sys") is None


def test_system_from_lines():
    lines = ["test_from_lines_sys", "[length]: meter", "[mass]: kilogram"]
    sys = System.from_lines(lines)
    assert sys.name == "test_from_lines_sys"
    assert sys.base_units == {"[length]": "meter", "[mass]": "kilogram"}


def test_system_repr():
    sys = System("test_sys_repr")
    sys.add_rule("[length]", "meter")
    assert "test_sys_repr" in repr(sys)
    assert "1" in repr(sys)


def test_system_convert():
    ureg = pintrs.UnitRegistry()
    cgs = ureg.get_system("cgs")
    val, unit = cgs.convert(ureg, 1.0, "meter")
    assert abs(val - 100.0) < 1e-10
    assert unit == "centimeter"


def test_system_convert_mass():
    ureg = pintrs.UnitRegistry()
    cgs = ureg.get_system("cgs")
    val, unit = cgs.convert(ureg, 1.0, "kilogram")
    assert abs(val - 1000.0) < 1e-10
    assert unit == "gram"


def test_system_convert_no_rule():
    ureg = pintrs.UnitRegistry()
    sys = System("test_sys_norule")
    val, unit = sys.convert(ureg, 42.0, "meter")
    assert val == 42.0
    assert unit == "meter"


def test_builtin_mks():
    ureg = pintrs.UnitRegistry()
    mks = ureg.get_system("mks")
    assert mks.preferred_unit_for("[length]") == "meter"
    assert mks.preferred_unit_for("[mass]") == "kilogram"
    assert mks.preferred_unit_for("[time]") == "second"


def test_builtin_si():
    ureg = pintrs.UnitRegistry()
    si = ureg.get_system("SI")
    assert si.preferred_unit_for("[length]") == "meter"
    assert si.preferred_unit_for("[mass]") == "kilogram"


def test_builtin_cgs():
    ureg = pintrs.UnitRegistry()
    cgs = ureg.get_system("cgs")
    assert cgs.preferred_unit_for("[length]") == "centimeter"
    assert cgs.preferred_unit_for("[mass]") == "gram"


def test_builtin_imperial():
    ureg = pintrs.UnitRegistry()
    imp = ureg.get_system("imperial")
    assert imp.preferred_unit_for("[length]") == "foot"
    assert imp.preferred_unit_for("[mass]") == "pound"


def test_builtin_atomic():
    ureg = pintrs.UnitRegistry()
    atomic = ureg.get_system("atomic")
    assert atomic.preferred_unit_for("[length]") == "bohr"
    assert atomic.preferred_unit_for("[mass]") == "electron_mass"


def test_get_unknown_system_creates_empty():
    ureg = pintrs.UnitRegistry()
    sys = ureg.get_system("brand_new_system")
    assert sys.name == "brand_new_system"
    assert sys.base_units == {}


def test_mks_convert_foot_to_meter():
    ureg = pintrs.UnitRegistry()
    mks = ureg.get_system("mks")
    val, unit = mks.convert(ureg, 1.0, "foot")
    assert abs(val - 0.3048) < 1e-10
    assert unit == "meter"
