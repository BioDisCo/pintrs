"""Comprehensive Unit tests - ported from pint's test suite."""

from __future__ import annotations

import copy
import pickle

import pytest
from pintrs import (
    DimensionalityError,
    UndefinedUnitError,
    Unit,
    UnitRegistry,
)


@pytest.fixture
def ureg() -> UnitRegistry:
    """Fresh registry for each test."""
    return UnitRegistry()


# ---------------------------------------------------------------------------
# 1. TestUnitCreation
# ---------------------------------------------------------------------------


class TestUnitCreation:
    """Unit creation from strings, parse_units, and compound expressions."""

    def test_create_from_string(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("meter")
        assert str(u) == "m"

    def test_create_from_symbol(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("m")
        assert str(u) == "m"

    def test_create_from_parse_units(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter")
        assert isinstance(u, Unit)
        assert str(u) == "m"

    def test_create_compound_mul(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter * second")
        assert "meter" in dict(u._units_dict())
        assert "second" in dict(u._units_dict())

    def test_create_compound_div(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter / second")
        d = dict(u._units_dict())
        assert d["meter"] == pytest.approx(1.0)
        assert d["second"] == pytest.approx(-1.0)

    def test_create_compound_pow(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter ** 2")
        d = dict(u._units_dict())
        assert d["meter"] == pytest.approx(2.0)

    def test_create_complex_compound(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("kilogram * meter / second ** 2")
        d = dict(u._units_dict())
        assert d["second"] == pytest.approx(-2.0)
        assert d["meter"] == pytest.approx(1.0)

    def test_create_dimensionless(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("dimensionless")
        assert u.dimensionless

    def test_create_prefixed(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("kilometer")
        assert str(u) in ("km", "kilometer")

    def test_create_from_abbreviation(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("kg")
        assert u.dimensionality == ureg.parse_units("kilogram").dimensionality

    def test_create_undefined_unit_raises(self, ureg: UnitRegistry) -> None:
        with pytest.raises(UndefinedUnitError):
            ureg.parse_units("nonexistent_xyzzy_unit")

    def test_unit_returns_unit_type(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("meter")
        assert type(u).__name__ == "Unit"

    def test_parse_units_returns_unit_type(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter")
        assert type(u).__name__ == "Unit"

    def test_getattr_returns_quantity(self, ureg: UnitRegistry) -> None:
        q = ureg.meter
        assert type(q).__name__ == "Quantity"
        assert q.magnitude == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 2. TestUnitArithmetic
# ---------------------------------------------------------------------------


class TestUnitArithmetic:
    """Multiplication, division, and exponentiation of units."""

    def test_mul_units(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter") * ureg.parse_units("second")
        assert isinstance(u, Unit)

    def test_div_units(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter") / ureg.parse_units("second")
        d = dict(u._units_dict())
        assert d["meter"] == pytest.approx(1.0)
        assert d["second"] == pytest.approx(-1.0)

    def test_pow_unit(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter") ** 3
        d = dict(u._units_dict())
        assert d["meter"] == pytest.approx(3.0)

    def test_pow_zero_gives_dimensionless(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter") ** 0
        assert u.dimensionless

    def test_pow_negative(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter") ** -1
        d = dict(u._units_dict())
        assert d["meter"] == pytest.approx(-1.0)

    def test_pow_fractional(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter") ** 0.5
        d = dict(u._units_dict())
        assert d["meter"] == pytest.approx(0.5)

    def test_mul_number_returns_quantity(self, ureg: UnitRegistry) -> None:
        result = ureg.parse_units("meter") * 3.0
        assert type(result).__name__ == "Quantity"
        assert result.magnitude == pytest.approx(3.0)

    def test_rmul_number_returns_quantity(self, ureg: UnitRegistry) -> None:
        result = 3.0 * ureg.parse_units("meter")
        assert type(result).__name__ == "Quantity"
        assert result.magnitude == pytest.approx(3.0)

    def test_compound_mul_then_div(self, ureg: UnitRegistry) -> None:
        m = ureg.parse_units("meter")
        s = ureg.parse_units("second")
        u = (m * m) / s
        d = dict(u._units_dict())
        assert d["meter"] == pytest.approx(2.0)
        assert d["second"] == pytest.approx(-1.0)

    def test_mul_dimensionless_identity(self, ureg: UnitRegistry) -> None:
        m = ureg.parse_units("meter")
        d = ureg.parse_units("dimensionless")
        result = m * d
        assert result == m


# ---------------------------------------------------------------------------
# 3. TestUnitComparison
# ---------------------------------------------------------------------------


class TestUnitComparison:
    """Equality, hashing, and compatibility checks."""

    def test_eq_same_unit(self, ureg: UnitRegistry) -> None:
        assert ureg.parse_units("meter") == ureg.parse_units("meter")

    def test_eq_symbol_and_name(self, ureg: UnitRegistry) -> None:
        assert ureg.parse_units("m") == ureg.parse_units("meter")

    def test_ne_different_units(self, ureg: UnitRegistry) -> None:
        assert ureg.parse_units("meter") != ureg.parse_units("second")

    def test_ne_different_dimension(self, ureg: UnitRegistry) -> None:
        assert ureg.parse_units("meter") != ureg.parse_units("kilogram")

    def test_ne_different_prefix(self, ureg: UnitRegistry) -> None:
        assert ureg.parse_units("meter") != ureg.parse_units("kilometer")

    def test_eq_non_unit_returns_false(self, ureg: UnitRegistry) -> None:
        assert ureg.parse_units("meter") != 42
        assert ureg.parse_units("meter") != "meter"

    def test_hash_equal_units(self, ureg: UnitRegistry) -> None:
        u1 = ureg.parse_units("meter")
        u2 = ureg.parse_units("meter")
        assert hash(u1) == hash(u2)

    def test_hash_different_units(self, ureg: UnitRegistry) -> None:
        u1 = ureg.parse_units("meter")
        u2 = ureg.parse_units("second")
        assert hash(u1) != hash(u2)

    def test_unit_in_set(self, ureg: UnitRegistry) -> None:
        s = {ureg.parse_units("meter"), ureg.parse_units("meter")}
        assert len(s) == 1

    def test_unit_as_dict_key(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter")
        d = {u: "length"}
        assert d[ureg.parse_units("meter")] == "length"

    def test_compatible_with_unit(self, ureg: UnitRegistry) -> None:
        m = ureg.parse_units("meter")
        km = ureg.parse_units("kilometer")
        assert m.is_compatible_with(km)

    def test_compatible_with_string(self, ureg: UnitRegistry) -> None:
        m = ureg.parse_units("meter")
        assert m.is_compatible_with("kilometer")

    def test_not_compatible_different_dimension(self, ureg: UnitRegistry) -> None:
        m = ureg.parse_units("meter")
        assert not m.is_compatible_with("second")

    def test_compatible_units_list(self, ureg: UnitRegistry) -> None:
        m = ureg.parse_units("meter")
        compat = m.compatible_units()
        assert isinstance(compat, list)
        assert len(compat) > 0
        assert "meter" in compat
        assert "foot" in compat

    def test_compound_units_equal(self, ureg: UnitRegistry) -> None:
        u1 = ureg.parse_units("meter / second")
        u2 = ureg.parse_units("meter") / ureg.parse_units("second")
        assert u1 == u2


# ---------------------------------------------------------------------------
# 4. TestUnitDimensionality
# ---------------------------------------------------------------------------


class TestUnitDimensionality:
    """Dimensionality strings and dimensionless property."""

    @pytest.mark.parametrize(
        ("unit_str", "expected_dim"),
        [
            ("meter", "[length]"),
            ("second", "[time]"),
            ("kilogram", "[mass]"),
            ("ampere", "[current]"),
            ("kelvin", "[temperature]"),
            ("mole", "[substance]"),
            ("candela", "[luminosity]"),
        ],
    )
    def test_base_unit_dimensionality(
        self,
        ureg: UnitRegistry,
        unit_str: str,
        expected_dim: str,
    ) -> None:
        u = ureg.parse_units(unit_str)
        assert u.dimensionality == expected_dim

    def test_compound_dimensionality(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter / second")
        assert "[length]" in u.dimensionality
        assert "[time]" in u.dimensionality

    def test_squared_dimensionality(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter ** 2")
        assert u.dimensionality == "[length] ** 2"

    def test_force_dimensionality(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("kilogram * meter / second ** 2")
        dim = u.dimensionality
        assert "[mass]" in dim
        assert "[length]" in dim
        assert "[time]" in dim

    def test_dimensionless_unit(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("dimensionless")
        assert u.dimensionless is True

    def test_not_dimensionless(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter")
        assert u.dimensionless is False

    def test_power_zero_is_dimensionless(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter") ** 0
        assert u.dimensionless is True

    def test_prefixed_unit_same_dimensionality(self, ureg: UnitRegistry) -> None:
        m = ureg.parse_units("meter")
        km = ureg.parse_units("kilometer")
        assert m.dimensionality == km.dimensionality


# ---------------------------------------------------------------------------
# 5. TestUnitRepresentation
# ---------------------------------------------------------------------------


class TestUnitRepresentation:
    """String, repr, and format specifier output."""

    def test_str(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter")
        assert str(u) == "m"

    def test_repr(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter")
        r = repr(u)
        assert "Unit" in r
        assert "m" in r

    def test_units_str(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter")
        assert u._units_str() == "m"

    def test_units_dict(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter")
        d = u._units_dict()
        assert ("meter", 1.0) in d

    def test_units_dict_compound(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter / second ** 2")
        d = dict(u._units_dict())
        assert d["meter"] == pytest.approx(1.0)
        assert d["second"] == pytest.approx(-2.0)

    def test_format_pretty(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter ** 2 / second ** 2")
        fmt = format(u, "P")
        assert "\u00b2" in fmt  # superscript 2

    def test_format_pretty_abbreviated(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter ** 2 / second ** 2")
        fmt = format(u, "~P")
        assert "m" in fmt
        assert "s" in fmt
        assert "\u00b2" in fmt

    def test_format_latex(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter ** 2 / second ** 2")
        fmt = format(u, "L")
        assert r"\frac" in fmt
        assert "$" in fmt

    def test_format_latex_abbreviated(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter ** 2 / second ** 2")
        fmt = format(u, "~L")
        assert r"\frac" in fmt

    def test_format_html(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter ** 2")
        fmt = format(u, "H")
        assert "<sup>" in fmt

    def test_format_compact(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter")
        fmt = format(u, "C")
        assert "meter" in fmt

    def test_format_default(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter")
        fmt = format(u, "D")
        assert "meter" in fmt

    def test_format_dimensionless(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("dimensionless")
        s = str(u)
        assert "dimensionless" in s.lower() or s == ""

    def test_str_compound(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter / second ** 2")
        s = str(u)
        assert "/" in s
        assert "2" in s


# ---------------------------------------------------------------------------
# 6. TestUnitSerialization
# ---------------------------------------------------------------------------


class TestUnitSerialization:
    """Pickle, copy, and deepcopy round-trips."""

    def test_pickle_simple(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter")
        data = pickle.dumps(u)
        u2 = pickle.loads(data)
        assert u == u2

    def test_pickle_compound(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("kilogram * meter / second ** 2")
        data = pickle.dumps(u)
        u2 = pickle.loads(data)
        assert u == u2

    def test_pickle_preserves_hash(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter")
        data = pickle.dumps(u)
        u2 = pickle.loads(data)
        assert hash(u) == hash(u2)

    def test_copy(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter")
        u2 = copy.copy(u)
        assert u == u2

    def test_deepcopy(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter")
        u2 = copy.deepcopy(u)
        assert u == u2

    def test_copy_compound(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter / second")
        u2 = copy.copy(u)
        assert u == u2

    def test_deepcopy_compound(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("kilogram * meter ** 2 / second ** 2")
        u2 = copy.deepcopy(u)
        assert u == u2

    def test_pickle_dimensionless(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("dimensionless")
        data = pickle.dumps(u)
        u2 = pickle.loads(data)
        assert u2.dimensionless


# ---------------------------------------------------------------------------
# 7. TestUnitConversion
# ---------------------------------------------------------------------------


class TestUnitConversion:
    """Unit.from_() and Unit.m_from() conversions."""

    def test_from_simple(self, ureg: UnitRegistry) -> None:
        meter = ureg.Unit("meter")
        q = ureg.Quantity(1, "kilometer")
        result = meter.from_(q)
        assert result.magnitude == pytest.approx(1000.0)

    def test_from_preserves_target_unit(self, ureg: UnitRegistry) -> None:
        meter = ureg.Unit("meter")
        q = ureg.Quantity(5, "kilometer")
        result = meter.from_(q)
        assert str(result.units) == "m"

    def test_m_from_simple(self, ureg: UnitRegistry) -> None:
        meter = ureg.Unit("meter")
        q = ureg.Quantity(1, "kilometer")
        assert meter.m_from(q) == pytest.approx(1000.0)

    def test_from_incompatible_raises(self, ureg: UnitRegistry) -> None:
        meter = ureg.Unit("meter")
        q = ureg.Quantity(1, "second")
        with pytest.raises(DimensionalityError):
            meter.from_(q)

    def test_m_from_incompatible_raises(self, ureg: UnitRegistry) -> None:
        meter = ureg.Unit("meter")
        q = ureg.Quantity(1, "second")
        with pytest.raises(DimensionalityError):
            meter.m_from(q)

    def test_from_same_unit(self, ureg: UnitRegistry) -> None:
        meter = ureg.Unit("meter")
        q = ureg.Quantity(5.0, "meter")
        result = meter.from_(q)
        assert result.magnitude == pytest.approx(5.0)

    def test_m_from_prefixed(self, ureg: UnitRegistry) -> None:
        mm = ureg.Unit("millimeter")
        q = ureg.Quantity(1, "meter")
        assert mm.m_from(q) == pytest.approx(1000.0)

    def test_from_compound_unit(self, ureg: UnitRegistry) -> None:
        ms = ureg.Unit("meter / second")
        q = ureg.Quantity(1, "kilometer / hour")
        result = ms.from_(q)
        assert result.magnitude == pytest.approx(1.0 / 3.6, rel=1e-6)

    def test_systems_property(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("meter")
        assert isinstance(u.systems, frozenset)
        assert "mks" in u.systems


# ---------------------------------------------------------------------------
# 8. TestRegistryUnitOps
# ---------------------------------------------------------------------------


class TestRegistryUnitOps:
    """Registry-level operations: define, get_name, get_symbol, etc."""

    def test_define_custom_unit(self, ureg: UnitRegistry) -> None:
        ureg.define("test_port_unit_a = 2 * meter")
        q = ureg.Quantity(1, "test_port_unit_a")
        result = q.to("meter")
        assert result.magnitude == pytest.approx(2.0)

    def test_define_with_alias(self, ureg: UnitRegistry) -> None:
        ureg.define("test_port_unit_b = 3 * meter = tpub")
        q = ureg.Quantity(1, "tpub")
        result = q.to("meter")
        assert result.magnitude == pytest.approx(3.0)

    def test_define_does_not_clobber_builtin(self, ureg: UnitRegistry) -> None:
        """Defining a unit that already exists should not break the registry."""
        original = ureg.Quantity(1, "meter").to("foot").magnitude
        ureg.define("test_port_shadow = 1 * meter")
        after = ureg.Quantity(1, "meter").to("foot").magnitude
        assert original == pytest.approx(after)

    def test_get_name(self, ureg: UnitRegistry) -> None:
        assert ureg.get_name("m") == "meter"

    def test_get_name_from_alias(self, ureg: UnitRegistry) -> None:
        name = ureg.get_name("metre")
        assert name == "meter"

    def test_get_symbol(self, ureg: UnitRegistry) -> None:
        sym = ureg.get_symbol("meter")
        assert sym == "m"

    def test_get_symbol_gram(self, ureg: UnitRegistry) -> None:
        sym = ureg.get_symbol("gram")
        assert sym == "g"

    def test_parse_unit_name_plain(self, ureg: UnitRegistry) -> None:
        result = ureg.parse_unit_name("meter")
        assert len(result) > 0
        assert result[0][1] == "meter"

    def test_parse_unit_name_prefixed(self, ureg: UnitRegistry) -> None:
        result = ureg.parse_unit_name("kilometer")
        assert len(result) > 0
        assert result[0][0] == "kilo"
        assert result[0][1] == "meter"

    def test_get_base_units(self, ureg: UnitRegistry) -> None:
        factor, base = ureg.get_base_units("kilometer")
        assert factor == pytest.approx(1000.0)
        assert isinstance(base, Unit)

    def test_get_base_units_compound(self, ureg: UnitRegistry) -> None:
        factor, _base = ureg.get_base_units("kilometer / hour")
        assert factor == pytest.approx(1000.0 / 3600.0, rel=1e-6)

    def test_get_compatible_units(self, ureg: UnitRegistry) -> None:
        compat = ureg.get_compatible_units("meter")
        assert isinstance(compat, list)
        assert "foot" in compat
        assert "inch" in compat

    def test_get_compatible_units_excludes_incompatible(
        self, ureg: UnitRegistry
    ) -> None:
        compat = ureg.get_compatible_units("meter")
        assert "second" not in compat
        assert "kilogram" not in compat

    def test_contains_known_unit(self, ureg: UnitRegistry) -> None:
        assert "meter" in ureg

    def test_contains_symbol(self, ureg: UnitRegistry) -> None:
        assert "m" in ureg

    def test_not_contains_unknown(self, ureg: UnitRegistry) -> None:
        assert "nonexistent_xyzzy_unit" not in ureg

    def test_iter_yields_unit_names(self, ureg: UnitRegistry) -> None:
        units = list(ureg)
        assert len(units) > 0
        assert "meter" in units
        assert "second" in units

    def test_load_definitions(self, ureg: UnitRegistry) -> None:
        ureg.load_definitions("test_port_loaded_unit = 42 * meter")
        assert "test_port_loaded_unit" in ureg
        q = ureg.Quantity(1, "test_port_loaded_unit")
        assert q.to("meter").magnitude == pytest.approx(42.0)

    def test_get_dimensionality(self, ureg: UnitRegistry) -> None:
        assert ureg.get_dimensionality("meter") == "[length]"

    def test_get_dimensionality_compound(self, ureg: UnitRegistry) -> None:
        dim = ureg.get_dimensionality("meter / second")
        assert "[length]" in dim
        assert "[time]" in dim

    def test_is_compatible_with_registry(self, ureg: UnitRegistry) -> None:
        assert ureg.is_compatible_with("meter", "foot")

    def test_is_not_compatible_with_registry(self, ureg: UnitRegistry) -> None:
        assert not ureg.is_compatible_with("meter", "second")

    def test_convert(self, ureg: UnitRegistry) -> None:
        result = ureg.convert(1.0, "kilometer", "meter")
        assert result == pytest.approx(1000.0)

    def test_parse_expression(self, ureg: UnitRegistry) -> None:
        q = ureg.parse_expression("5 meter")
        assert q.magnitude == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# 9. TestUnitParsing
# ---------------------------------------------------------------------------


class TestUnitParsing:
    """Parsing of simple, prefixed, compound, plural, and alias forms."""

    @pytest.mark.parametrize(
        "unit_str",
        ["meter", "m", "metre", "meters", "metres"],
    )
    def test_meter_aliases(self, ureg: UnitRegistry, unit_str: str) -> None:
        u = ureg.parse_units(unit_str)
        assert u == ureg.parse_units("meter")

    @pytest.mark.parametrize(
        ("prefix", "base"),
        [
            ("kilo", "meter"),
            ("milli", "meter"),
            ("centi", "meter"),
            ("micro", "meter"),
            ("nano", "meter"),
        ],
    )
    def test_prefixed_units(
        self,
        ureg: UnitRegistry,
        prefix: str,
        base: str,
    ) -> None:
        u = ureg.parse_units(f"{prefix}{base}")
        assert u.dimensionality == ureg.parse_units(base).dimensionality

    @pytest.mark.parametrize(
        "expr",
        [
            "meter / second",
            "meter * second",
            "kilogram * meter / second ** 2",
            "meter ** 2",
            "meter ** 3",
        ],
    )
    def test_compound_expressions(self, ureg: UnitRegistry, expr: str) -> None:
        u = ureg.parse_units(expr)
        assert len(u._units_dict()) > 0

    def test_parse_newton(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("newton")
        dim = u.dimensionality
        assert "[mass]" in dim
        assert "[length]" in dim
        assert "[time]" in dim

    def test_parse_joule(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("joule")
        dim = u.dimensionality
        assert "[mass]" in dim
        assert "[length]" in dim

    def test_parse_watt(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("watt")
        dim = u.dimensionality
        assert "[mass]" in dim
        assert "[time]" in dim

    def test_parse_temperature_celsius(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("degree_Celsius")
        assert "[temperature]" in u.dimensionality

    def test_parse_temperature_fahrenheit(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("degree_Fahrenheit")
        assert "[temperature]" in u.dimensionality

    def test_parse_round_trip_simple(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter")
        s = str(u)
        u2 = ureg.parse_units(s)
        assert u == u2

    def test_parse_round_trip_compound(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter / second ** 2")
        s = str(u)
        u2 = ureg.parse_units(s)
        assert u == u2

    def test_parse_unit_with_per(self, ureg: UnitRegistry) -> None:
        u = ureg.parse_units("meter / second")
        d = dict(u._units_dict())
        assert d["second"] == pytest.approx(-1.0)

    @pytest.mark.parametrize(
        "unit_str",
        [
            "foot",
            "inch",
            "mile",
            "yard",
            "pound",
            "ounce",
        ],
    )
    def test_imperial_units_exist(self, ureg: UnitRegistry, unit_str: str) -> None:
        u = ureg.parse_units(unit_str)
        assert len(u._units_dict()) > 0
