"""Tests for pint API compatibility.

These verify that pintrs exposes the same API surface as pint,
so that libraries using pint can use pintrs as a drop-in replacement.
"""

from __future__ import annotations

import numpy as np
import pytest
from pintrs import Quantity, UnitRegistry
from pintrs._core import Quantity as _RustQuantity
from pintrs.numpy_support import ArrayQuantity

# --- Quantity.to() / m_as() accept Unit and Quantity arguments ---


class TestToAcceptsNonString:
    def test_to_accepts_unit_object(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1000.0, "m")
        result = q.to(ureg.Unit("km"))
        assert result.magnitude == pytest.approx(1.0)

    def test_to_accepts_quantity_as_units(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1000.0, "m")
        result = q.to(1 * ureg.km)
        assert result.magnitude == pytest.approx(1.0)

    def test_m_as_accepts_unit_object(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1000.0, "m")
        assert q.m_as(ureg.Unit("km")) == pytest.approx(1.0)

    def test_m_as_accepts_quantity(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1000.0, "m")
        assert q.m_as(1 * ureg.km) == pytest.approx(1.0)

    def test_ito_accepts_unit_object(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1000.0, "m")
        q.ito(ureg.Unit("km"))
        assert q.magnitude == pytest.approx(1.0)

    def test_array_to_accepts_unit_object(self, ureg: UnitRegistry) -> None:
        v = np.array([1000.0, 2000.0]) * ureg.m
        result = v.to(ureg.Unit("km"))
        np.testing.assert_allclose(result.magnitude, [1.0, 2.0])

    def test_array_m_as_accepts_unit_object(self, ureg: UnitRegistry) -> None:
        v = np.array([1000.0]) * ureg.m
        result = v.m_as(ureg.Unit("km"))
        np.testing.assert_allclose(result, [1.0])


# --- isinstance checks ---


class TestInstanceCheck:
    def test_scalar_is_quantity(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "m")
        assert isinstance(q, Quantity)

    def test_array_is_quantity(self, ureg: UnitRegistry) -> None:
        v = np.array([1.0, 2.0]) * ureg.m
        assert isinstance(v, Quantity)

    def test_array_after_arithmetic_is_quantity(self, ureg: UnitRegistry) -> None:
        v = (np.array([3.0, 4.0]) * ureg.m) / (2.0 * ureg.s)
        assert isinstance(v, Quantity)

    def test_scalar_is_rust_quantity(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "m")
        assert isinstance(q, _RustQuantity)

    def test_array_is_array_quantity(self, ureg: UnitRegistry) -> None:
        v = np.array([1.0]) * ureg.m
        assert isinstance(v, ArrayQuantity)


# --- ArrayQuantity methods ---


class TestArrayQuantityMethods:
    def test_sum(self, ureg: UnitRegistry) -> None:
        v = np.array([1.0, 2.0, 3.0]) * ureg.m
        result = v.sum()
        assert isinstance(result, Quantity)
        assert result.magnitude == pytest.approx(6.0)

    def test_sum_preserves_units(self, ureg: UnitRegistry) -> None:
        v = np.array([1.0, 1.0]) * ureg.kg
        result = v.sum()
        assert result.dimensionality == {"[mass]": 1}

    def test_mean(self, ureg: UnitRegistry) -> None:
        v = np.array([2.0, 4.0]) * ureg.m
        result = v.mean()
        assert isinstance(result, Quantity)
        assert result.magnitude == pytest.approx(3.0)

    def test_reshape(self, ureg: UnitRegistry) -> None:
        v = np.array([1.0, 2.0, 3.0, 4.0]) * ureg.m
        result = v.reshape((2, 2))
        assert result.shape == (2, 2)
        np.testing.assert_allclose(result.magnitude, [[1.0, 2.0], [3.0, 4.0]])

    def test_numpy_sqrt(self, ureg: UnitRegistry) -> None:
        v = np.array([4.0, 9.0]) * ureg.m**2
        result = np.sqrt(v)
        assert isinstance(result, Quantity)
        np.testing.assert_allclose(result.magnitude, [2.0, 3.0])

    def test_numpy_sqrt_units(self, ureg: UnitRegistry) -> None:
        v = np.array([4.0]) * ureg.m**2
        result = np.sqrt(v)
        assert result.dimensionality == {"[length]": 1}


# --- ArrayQuantity.dimensionality returns dict-like ---


class TestArrayDimensionality:
    def test_returns_dict_like(self, ureg: UnitRegistry) -> None:
        v = np.array([1.0]) * ureg.m
        dim = v.dimensionality
        assert dict(dim) == {"[length]": 1.0}

    def test_equality_with_dict(self, ureg: UnitRegistry) -> None:
        v = np.array([1.0]) * ureg.kg
        assert v.dimensionality == {"[mass]": 1}

    def test_dimensionless(self, ureg: UnitRegistry) -> None:
        v = np.array([1.0]) * ureg.m / ureg.m
        assert dict(v.dimensionality) == {}


# --- _REGISTRY identity ---


class TestRegistryIdentity:
    def test_scalar_registry_is_creator(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(6.0, "m")
        assert q._REGISTRY is ureg

    def test_scalar_registry_stable(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "s")
        assert q._REGISTRY is q._REGISTRY

    def test_getattr_quantity_registry(self, ureg: UnitRegistry) -> None:
        u = ureg.m
        assert u._REGISTRY is not None

    def test_array_registry_is_creator(self, ureg: UnitRegistry) -> None:
        q = np.array([1.0, 2.0]) * ureg.m
        assert q._REGISTRY is not None

    def test_f64_quantity_registry(self, ureg: UnitRegistry) -> None:
        q = ureg._f64_quantity(3.14, "m")
        assert q._REGISTRY is ureg


# --- Kilogram is SI base unit for mass ---


class TestKilogramBase:
    def test_kg_is_base(self, ureg: UnitRegistry) -> None:
        factor, unit = ureg.get_base_units("kilogram")
        assert factor == pytest.approx(1.0)
        assert str(unit) == "kilogram"

    def test_gram_factor(self, ureg: UnitRegistry) -> None:
        factor, _ = ureg.get_base_units("gram")
        assert factor == pytest.approx(0.001)

    def test_mg_factor(self, ureg: UnitRegistry) -> None:
        factor, _ = ureg.get_base_units("milligram")
        assert factor == pytest.approx(1e-6)

    def test_to_base_units_kg(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(2.0, "kg")
        base = q.to_base_units()
        assert base.magnitude == pytest.approx(2.0)
        assert str(base.units) == "kilogram"

    def test_to_base_units_gram(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1000.0, "g")
        base = q.to_base_units()
        assert base.magnitude == pytest.approx(1.0)
        assert str(base.units) == "kilogram"


# --- Unit._REGISTRY ---


class TestUnitRegistry:
    def test_unit_has_registry(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("meter")
        assert u._REGISTRY is not None

    def test_unit_registry_attribute(self, ureg: UnitRegistry) -> None:
        u = ureg.Unit("meter")
        assert hasattr(u, "_REGISTRY")
        assert hasattr(u, "_registry")
