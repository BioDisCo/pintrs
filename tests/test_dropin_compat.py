"""Drop-in compatibility tests for third-party pint-using libraries.

Each test class exercises the pint API surface that a specific library depends on,
without requiring that library to be installed. These are regression tests for
compatibility fixes -- if a test here fails, the corresponding library will break.
"""

from __future__ import annotations

import numpy as np
import pytest
from pintrs import Quantity, Unit, UnitRegistry

# ---------------------------------------------------------------------------
# unit-jit (https://github.com/BioDisCo/unit-jit)
# ---------------------------------------------------------------------------


class TestUnitJitCompat:
    """APIs used by unit-jit's abstract interpreter and boundary conversion."""

    def test_quantity_to_base_units(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(100.0, "cm")
        base = q.to_base_units()
        assert base.magnitude == pytest.approx(1.0)
        assert str(base.units) == "m"

    def test_quantity_to_base_units_magnitude(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(2.0, "kg")
        assert q.to_base_units().magnitude == pytest.approx(2.0)

    def test_quantity_to_base_units_units(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "km")
        base = q.to_base_units()
        assert str(base.units) == "m"

    def test_quantity_dimensionality_dict_like(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "m/s")
        dim = q.dimensionality
        assert dict(dim) == {"[length]": 1.0, "[time]": -1.0}

    def test_quantity_dimensionality_equality(self, ureg: UnitRegistry) -> None:
        q1 = ureg.Quantity(1.0, "m")
        q2 = ureg.Quantity(1.0, "km")
        assert q1.dimensionality == q2.dimensionality

    def test_quantity_dimensionality_inequality(self, ureg: UnitRegistry) -> None:
        q1 = ureg.Quantity(1.0, "m")
        q2 = ureg.Quantity(1.0, "s")
        assert q1.dimensionality != q2.dimensionality

    def test_quantity_registry_attribute(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "m")
        assert q._REGISTRY is not None

    def test_unit_registry_attribute(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("m")
        assert u._REGISTRY is not None

    def test_unit_dimensionality(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("m/s**2")
        dim = u.dimensionality
        assert dict(dim) == {"[length]": 1.0, "[time]": -2.0}

    def test_unit_arithmetic_mul(self, ureg: UnitRegistry) -> None:
        u1 = ureg.Unit("m")
        u2 = ureg.Unit("s")
        result = u1 * u2
        assert "[length]" in str(result.dimensionality)

    def test_unit_arithmetic_div(self, ureg: UnitRegistry) -> None:
        u1 = ureg.Unit("m")
        u2 = ureg.Unit("s")
        result = u1 / u2
        dim = dict(result.dimensionality)
        assert dim["[length]"] == pytest.approx(1.0)
        assert dim["[time]"] == pytest.approx(-1.0)

    def test_unit_arithmetic_pow(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("m")
        result = u**2
        assert dict(result.dimensionality) == {"[length]": 2.0}

    def test_isinstance_check(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "m")
        assert isinstance(q, Quantity)

    def test_isinstance_check_array(self, ureg: UnitRegistry) -> None:
        v = np.array([1.0, 2.0]) * ureg.m
        assert isinstance(v, Quantity)

    def test_ureg_getattr_returns_unit(self, ureg: UnitRegistry) -> None:
        u = ureg.m
        assert isinstance(u, Unit)
        assert str(u) == "m"

    def test_ureg_quantity_constructor(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "m/s")
        assert q.magnitude == pytest.approx(5.0)

    def test_isinstance_unit_registry(self, ureg: UnitRegistry) -> None:
        assert isinstance(ureg, UnitRegistry)


# ---------------------------------------------------------------------------
# PyMeasure (https://github.com/pymeasure/pymeasure)
# ---------------------------------------------------------------------------


class TestPyMeasureCompat:
    """APIs used by PyMeasure for instrument parameter handling."""

    def test_m_as_with_unit_object(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.5, "m")
        assert q.m_as(ureg.Unit("m")) == pytest.approx(5.5)

    def test_m_as_with_unit_conversion(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1000.0, "mm")
        assert q.m_as(ureg.Unit("m")) == pytest.approx(1.0)

    def test_quantity_creation_from_string(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "V")
        assert str(q.units) in ("V", "volt")

    def test_quantity_is_compatible(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "m")
        assert q.is_compatible_with("km")
        assert not q.is_compatible_with("s")


# ---------------------------------------------------------------------------
# fluids (https://github.com/CalebBell/fluids)
# ---------------------------------------------------------------------------


class TestFluidsCompat:
    """APIs used by fluids' wraps_numpydoc unit wrapper."""

    def test_to_accepts_quantity(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "m")
        result = q.to(ureg.m)
        assert result.magnitude == pytest.approx(1.0)

    def test_to_accepts_unit(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(100.0, "cm")
        result = q.to(ureg.Unit("m"))
        assert result.magnitude == pytest.approx(1.0)

    def test_array_dimensionality_dict_like(self, ureg: UnitRegistry) -> None:
        v = np.array([1.0]) * ureg.m
        assert dict(v.dimensionality) == {"[length]": 1.0}

    def test_array_dimensionality_equality(self, ureg: UnitRegistry) -> None:
        v = np.array([1.0]) * ureg.m
        q = ureg.Quantity(1.0, "m")
        assert v.dimensionality == q.dimensionality


# ---------------------------------------------------------------------------
# pint-pandas (https://github.com/hgrecco/pint-pandas)
# ---------------------------------------------------------------------------


class TestPintPandasCompat:
    """APIs used by pint-pandas' PintArray/PintDtype."""

    def test_to_with_unit_object(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "m")
        result = q.to(ureg.Unit("km"))
        assert result.magnitude == pytest.approx(0.001)

    def test_array_to_with_unit_object(self, ureg: UnitRegistry) -> None:
        v = np.array([1000.0, 2000.0]) * ureg.m
        result = v.to(ureg.Unit("km"))
        np.testing.assert_allclose(result.magnitude, [1.0, 2.0])

    def test_registry_identity_from_quantity(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "m")
        assert q._REGISTRY is ureg

    def test_registry_identity_from_getattr(self, ureg: UnitRegistry) -> None:
        u = ureg.m
        assert isinstance(u, Unit)
        assert u._REGISTRY is not None

    def test_registry_identity_from_array(self, ureg: UnitRegistry) -> None:
        v = np.array([1.0]) * ureg.m
        assert v._REGISTRY is not None


# ---------------------------------------------------------------------------
# pint-xarray (https://github.com/xarray-contrib/pint-xarray)
# ---------------------------------------------------------------------------


class TestPintXarrayCompat:
    """APIs used by pint-xarray for quantity-wrapped DataArrays."""

    def test_array_sum(self, ureg: UnitRegistry) -> None:
        v = np.array([1.0, 2.0, 3.0]) * ureg.m
        result = v.sum()
        assert result.magnitude == pytest.approx(6.0)

    def test_array_mean(self, ureg: UnitRegistry) -> None:
        v = np.array([2.0, 4.0]) * ureg.m
        result = v.mean()
        assert result.magnitude == pytest.approx(3.0)

    def test_array_reshape(self, ureg: UnitRegistry) -> None:
        v = np.array([1.0, 2.0, 3.0, 4.0]) * ureg.m
        result = v.reshape((2, 2))
        assert result.shape == (2, 2)

    def test_numpy_sqrt_on_array_quantity(self, ureg: UnitRegistry) -> None:
        v = np.array([4.0, 9.0]) * ureg.m**2
        result = np.sqrt(v)
        np.testing.assert_allclose(result.magnitude, [2.0, 3.0])
        assert result.dimensionality == {"[length]": 1}


# ---------------------------------------------------------------------------
# MetPy (https://github.com/Unidata/MetPy)
# ---------------------------------------------------------------------------


class TestMetPyCompat:
    """APIs used by MetPy's @check_units decorator and unit handling."""

    def test_array_dimensionality_is_dict_subclass(self, ureg: UnitRegistry) -> None:
        v = np.array([1.0]) * ureg.m
        dim = v.dimensionality
        assert isinstance(dim, dict)

    def test_scalar_dimensionality_is_dict_subclass(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "m")
        dim = q.dimensionality
        assert isinstance(dim, dict)

    def test_dimensionality_cross_type_equality(self, ureg: UnitRegistry) -> None:
        """Scalar and array dimensionality must compare equal."""
        q = ureg.Quantity(1.0, "m")
        v = np.array([1.0]) * ureg.m
        assert q.dimensionality == v.dimensionality

    def test_quantity_constructor_with_unit_object(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("m/s")
        q = ureg.Quantity(5.0, str(u))
        assert q.magnitude == pytest.approx(5.0)

    def test_get_dimensionality(self, ureg: UnitRegistry) -> None:
        dim = ureg.get_dimensionality("m / s ** 2")
        assert "[length]" in dim
        assert "[time]" in dim


# ---------------------------------------------------------------------------
# pyam (https://github.com/IAMconsortium/pyam)
# ---------------------------------------------------------------------------


class TestPyamCompat:
    """APIs used by pyam's unit conversion pipeline."""

    def test_quantity_convert(self, ureg: UnitRegistry) -> None:
        factor = ureg.convert(1.0, "km", "m")
        assert factor == pytest.approx(1000.0)

    def test_parse_expression(self, ureg: UnitRegistry) -> None:
        q = ureg.parse_expression("5.0 m")
        assert q.magnitude == pytest.approx(5.0)

    def test_array_to_string(self, ureg: UnitRegistry) -> None:
        v = np.array([1.0, 2.0]) * ureg.m
        result = v.to("km")
        np.testing.assert_allclose(result.magnitude, [0.001, 0.002])

    def test_define_new_unit(self, ureg: UnitRegistry) -> None:
        ureg.define("testunit = 42 * meter")
        q = ureg.Quantity(1.0, "testunit")
        assert q.to("m").magnitude == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# Kilogram base unit (affects all mass-using libraries)
# ---------------------------------------------------------------------------


class TestKilogramBase:
    """Kilogram must be the SI base unit for mass, not gram."""

    def test_kg_is_base(self, ureg: UnitRegistry) -> None:
        factor, unit = ureg.get_base_units("kilogram")
        assert factor == pytest.approx(1.0)
        assert str(unit) == "kilogram"

    def test_gram_factor(self, ureg: UnitRegistry) -> None:
        factor, _ = ureg.get_base_units("gram")
        assert factor == pytest.approx(0.001)

    def test_to_base_units_preserves_kg(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(2.0, "kg")
        base = q.to_base_units()
        assert base.magnitude == pytest.approx(2.0)
        assert str(base.units) == "kilogram"

    def test_to_base_units_converts_g_to_kg(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1000.0, "g")
        base = q.to_base_units()
        assert base.magnitude == pytest.approx(1.0)
        assert str(base.units) == "kilogram"

    def test_newton_definition(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "N")
        base = q.to_base_units()
        assert str(base.units) == "kilogram * m / s ** 2"
        assert base.magnitude == pytest.approx(1.0)
