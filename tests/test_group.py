"""Tests for pintrs Group functionality."""

from __future__ import annotations

import pintrs
from pintrs.group import Group


def test_group_creation():
    grp = Group("test_group_1")
    assert grp.name == "test_group_1"
    assert len(grp.members) == 0


def test_group_add_remove_units():
    grp = Group("test_group_2")
    grp.add_units("meter", "foot", "inch")
    assert "meter" in grp
    assert "foot" in grp
    assert len(grp) == 3

    grp.remove_units("foot")
    assert "foot" not in grp
    assert len(grp) == 2


def test_group_members_frozenset():
    grp = Group("test_group_3")
    grp.add_units("meter", "foot")
    members = grp.members
    assert isinstance(members, frozenset)
    assert members == frozenset({"meter", "foot"})


def test_group_used_groups():
    parent = Group("test_parent")
    parent.add_units("meter", "kilogram")

    child = Group("test_child")
    child.add_units("second")
    child.add_used_groups("test_parent")

    assert "meter" in child
    assert "kilogram" in child
    assert "second" in child
    assert len(child) == 3


def test_group_remove_used_groups():
    parent = Group("test_parent2")
    parent.add_units("meter")

    child = Group("test_child2")
    child.add_units("second")
    child.add_used_groups("test_parent2")
    assert "meter" in child

    child.remove_used_groups("test_parent2")
    assert "meter" not in child
    assert "second" in child


def test_group_registry_lookup():
    grp = Group("test_lookup")
    grp.add_units("meter")
    assert Group.get("test_lookup") is grp
    assert Group.get("nonexistent") is None


def test_group_from_lines():
    lines = ["test_from_lines", "meter foot", "kilogram"]
    grp = Group.from_lines(lines)
    assert grp.name == "test_from_lines"
    assert "meter" in grp
    assert "foot" in grp
    assert "kilogram" in grp


def test_group_iteration():
    grp = Group("test_iter")
    grp.add_units("a", "b", "c")
    items = set(grp)
    assert items == {"a", "b", "c"}


def test_group_repr():
    grp = Group("test_repr")
    grp.add_units("meter", "foot")
    assert "test_repr" in repr(grp)
    assert "2" in repr(grp)


def test_builtin_imperial_group():
    ureg = pintrs.UnitRegistry()
    imperial = ureg.get_group("imperial")
    assert "foot" in imperial
    assert "inch" in imperial
    assert "yard" in imperial
    assert "mile" in imperial
    assert "pound" in imperial
    assert "ounce" in imperial
    assert "meter" not in imperial


def test_builtin_metric_group():
    ureg = pintrs.UnitRegistry()
    metric = ureg.get_group("metric")
    assert "meter" in metric
    assert "gram" in metric
    assert "second" in metric
    assert "kelvin" in metric
    assert "foot" not in metric


def test_builtin_root_group():
    ureg = pintrs.UnitRegistry()
    root = ureg.get_group("root")
    assert "meter" in root
    assert "foot" in root
    assert "gram" in root
    assert len(root) > 100


def test_builtin_us_customary_group():
    ureg = pintrs.UnitRegistry()
    us = ureg.get_group("US_customary")
    # US_customary includes imperial units
    assert "foot" in us
    assert "pound" in us


def test_builtin_cgs_group():
    ureg = pintrs.UnitRegistry()
    cgs = ureg.get_group("cgs")
    assert "gram" in cgs
    assert "second" in cgs
    assert "dyne" in cgs
    assert "erg" in cgs


def test_get_unknown_group_creates_empty():
    ureg = pintrs.UnitRegistry()
    grp = ureg.get_group("brand_new_group")
    assert grp.name == "brand_new_group"
    assert len(grp) == 0
