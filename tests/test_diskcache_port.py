"""Diskcache integration tests - ported from pint's test_diskcache.py.

Tests that pintrs Quantity/Unit objects can be cached via diskcache.
"""

from __future__ import annotations

import pytest
from pintrs import UnitRegistry

pytest.importorskip("diskcache", reason="diskcache is not available")
import diskcache


@pytest.fixture
def ureg() -> UnitRegistry:
    return UnitRegistry()


class TestDiskcacheQuantity:
    """Test caching pintrs Quantity objects with diskcache."""

    def test_cache_scalar_quantity(self, ureg: UnitRegistry, tmp_path: object) -> None:
        cache = diskcache.Cache(str(tmp_path))
        q = ureg.Quantity(42.0, "meter")
        cache["distance"] = q
        retrieved = cache["distance"]
        assert retrieved.magnitude == pytest.approx(42.0)
        assert str(retrieved.units) == str(q.units)
        cache.close()

    def test_cache_converted_quantity(
        self,
        ureg: UnitRegistry,
        tmp_path: object,
    ) -> None:
        cache = diskcache.Cache(str(tmp_path))
        q = ureg.Quantity(1000, "meter").to("kilometer")
        cache["dist_km"] = q
        retrieved = cache["dist_km"]
        assert retrieved.magnitude == pytest.approx(1.0)
        cache.close()

    def test_cache_multiple_quantities(
        self,
        ureg: UnitRegistry,
        tmp_path: object,
    ) -> None:
        cache = diskcache.Cache(str(tmp_path))
        cache["mass"] = ureg.Quantity(5.0, "kilogram")
        cache["time"] = ureg.Quantity(10.0, "second")
        cache["speed"] = ureg.Quantity(3.0, "meter / second")

        assert cache["mass"].magnitude == pytest.approx(5.0)
        assert cache["time"].magnitude == pytest.approx(10.0)
        assert cache["speed"].magnitude == pytest.approx(3.0)
        cache.close()

    def test_cache_overwrite(self, ureg: UnitRegistry, tmp_path: object) -> None:
        cache = diskcache.Cache(str(tmp_path))
        cache["val"] = ureg.Quantity(1.0, "meter")
        cache["val"] = ureg.Quantity(2.0, "meter")
        assert cache["val"].magnitude == pytest.approx(2.0)
        cache.close()

    def test_cache_no_cache_folder(self) -> None:
        ureg = UnitRegistry(cache_folder=None)
        assert ureg.cache_folder is None


class TestDiskcacheUnit:
    """Test caching pintrs Unit objects with diskcache."""

    def test_cache_unit(self, ureg: UnitRegistry, tmp_path: object) -> None:
        cache = diskcache.Cache(str(tmp_path))
        u = ureg.Unit("meter / second ** 2")
        cache["accel_unit"] = u
        retrieved = cache["accel_unit"]
        assert str(retrieved) == str(u)
        cache.close()
