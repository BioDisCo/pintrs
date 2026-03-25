"""xarray integration tests.

Tests pintrs Quantity/ArrayQuantity interop with xarray DataArrays.
"""

from __future__ import annotations

import pytest
from pintrs import UnitRegistry
from pintrs.numpy_support import ArrayQuantity

np = pytest.importorskip("numpy", reason="numpy is not available")
xr = pytest.importorskip("xarray", reason="xarray is not available")


@pytest.fixture
def ureg() -> UnitRegistry:
    return UnitRegistry()


class TestXarrayInterop:
    """Test pintrs quantities with xarray data structures."""

    def test_dataarray_from_quantity_magnitude(self, ureg: UnitRegistry) -> None:
        q = ArrayQuantity(np.arange(5.0), "meter", ureg)
        da = xr.DataArray(q.magnitude, dims=["x"])
        assert da.shape == (5,)
        assert float(da[0]) == pytest.approx(0.0)

    def test_quantity_from_dataarray_values(self, ureg: UnitRegistry) -> None:
        da = xr.DataArray([1.0, 2.0, 3.0], dims=["x"])
        q = ArrayQuantity(da.values, "kelvin", ureg)
        assert q.magnitude[1] == pytest.approx(2.0)

    def test_dataarray_operations_then_quantity(self, ureg: UnitRegistry) -> None:
        da = xr.DataArray([10.0, 20.0, 30.0], dims=["x"])
        result = (da * 2).values
        q = ArrayQuantity(result, "second", ureg)
        assert q.magnitude[2] == pytest.approx(60.0)

    def test_dataset_with_quantity_data(self, ureg: UnitRegistry) -> None:
        temp = ArrayQuantity(np.array([300.0, 310.0, 320.0]), "kelvin", ureg)
        pressure = ArrayQuantity(np.array([101.0, 102.0, 103.0]), "kilopascal", ureg)
        ds = xr.Dataset(
            {
                "temperature": xr.DataArray(temp.magnitude, dims=["station"]),
                "pressure": xr.DataArray(pressure.magnitude, dims=["station"]),
            }
        )
        assert ds["temperature"].shape == (3,)
        assert ds["pressure"].shape == (3,)

    def test_xarray_reduction_to_quantity(self, ureg: UnitRegistry) -> None:
        da = xr.DataArray([1.0, 2.0, 3.0, 4.0, 5.0], dims=["t"])
        mean_val = float(da.mean())
        q = ureg.Quantity(mean_val, "meter / second")
        assert q.magnitude == pytest.approx(3.0)

    def test_xarray_coords_from_quantity(self, ureg: UnitRegistry) -> None:
        times = ArrayQuantity(np.arange(0, 10.0), "second", ureg)
        values = ArrayQuantity(np.random.default_rng(42).random(10), "meter", ureg)
        da = xr.DataArray(
            values.magnitude,
            coords={"time": times.magnitude},
            dims=["time"],
        )
        assert da.shape == (10,)
        assert float(da.coords["time"][0]) == pytest.approx(0.0)

    def test_xarray_sel_then_quantity(self, ureg: UnitRegistry) -> None:
        da = xr.DataArray(
            [100.0, 200.0, 300.0],
            coords={"x": [0, 1, 2]},
            dims=["x"],
        )
        val = float(da.sel(x=1))
        q = ureg.Quantity(val, "pascal")
        assert q.magnitude == pytest.approx(200.0)

    def test_2d_dataarray_quantity_roundtrip(self, ureg: UnitRegistry) -> None:
        data = np.arange(12.0).reshape(3, 4)
        q = ArrayQuantity(data, "watt", ureg)
        da = xr.DataArray(q.magnitude, dims=["row", "col"])
        q2 = ArrayQuantity(da.values, "watt", ureg)
        assert np.array_equal(q.magnitude, q2.magnitude)
