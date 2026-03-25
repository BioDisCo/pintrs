"""Tests for pandas ExtensionArray integration."""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from pintrs import UnitRegistry
from pintrs.pandas_support import PintArray, PintDtype


@pytest.fixture
def ureg() -> UnitRegistry:
    return UnitRegistry()


class TestPintDtype:
    def test_name(self) -> None:
        dtype = PintDtype("meter")
        assert dtype.name == "pint[meter]"

    def test_from_string(self) -> None:
        dtype = PintDtype.construct_from_string("pint[meter]")
        assert dtype.units == "meter"

    def test_invalid_string(self) -> None:
        with pytest.raises(TypeError):
            PintDtype.construct_from_string("int64")

    def test_eq(self) -> None:
        assert PintDtype("meter") == PintDtype("meter")
        assert PintDtype("meter") != PintDtype("second")

    def test_repr(self) -> None:
        dtype = PintDtype("meter")
        assert "meter" in repr(dtype)


class TestPintArray:
    def test_create(self, ureg: UnitRegistry) -> None:
        dtype = PintDtype("meter", ureg)
        arr = PintArray([1.0, 2.0, 3.0], dtype=dtype)
        assert len(arr) == 3

    def test_getitem_scalar(self, ureg: UnitRegistry) -> None:
        dtype = PintDtype("meter", ureg)
        arr = PintArray([1.0, 2.0, 3.0], dtype=dtype)
        item = arr[0]
        assert item.magnitude == 1.0

    def test_getitem_slice(self, ureg: UnitRegistry) -> None:
        dtype = PintDtype("meter", ureg)
        arr = PintArray([1.0, 2.0, 3.0], dtype=dtype)
        sliced = arr[0:2]
        assert isinstance(sliced, PintArray)
        assert len(sliced) == 2

    def test_setitem(self, ureg: UnitRegistry) -> None:
        dtype = PintDtype("meter", ureg)
        arr = PintArray([1.0, 2.0, 3.0], dtype=dtype)
        arr[1] = 5.0
        assert arr[1].magnitude == 5.0

    def test_isna(self, ureg: UnitRegistry) -> None:
        dtype = PintDtype("meter", ureg)
        arr = PintArray([1.0, float("nan"), 3.0], dtype=dtype)
        na_mask = arr.isna()
        assert not na_mask[0]
        assert na_mask[1]
        assert not na_mask[2]

    def test_to_conversion(self, ureg: UnitRegistry) -> None:
        dtype = PintDtype("kilometer", ureg)
        arr = PintArray([1.0, 2.0], dtype=dtype)
        result = arr.to("meter")
        assert abs(result[0].magnitude - 1000.0) < 1e-10
        assert result.units == "meter"

    def test_copy(self, ureg: UnitRegistry) -> None:
        dtype = PintDtype("meter", ureg)
        arr = PintArray([1.0, 2.0], dtype=dtype)
        arr2 = arr.copy()
        arr2[0] = 99.0
        assert arr[0].magnitude == 1.0

    def test_concat(self, ureg: UnitRegistry) -> None:
        dtype = PintDtype("meter", ureg)
        a = PintArray([1.0, 2.0], dtype=dtype)
        b = PintArray([3.0, 4.0], dtype=dtype)
        result = PintArray._concat_same_type([a, b])
        assert len(result) == 4


class TestPandasIntegration:
    def test_series_creation(self, ureg: UnitRegistry) -> None:
        dtype = PintDtype("meter", ureg)
        s = pd.Series(PintArray([1.0, 2.0, 3.0], dtype=dtype))
        assert len(s) == 3

    def test_dataframe_column(self, ureg: UnitRegistry) -> None:
        dtype = PintDtype("meter", ureg)
        df = pd.DataFrame({"distance": PintArray([1.0, 2.0], dtype=dtype)})
        assert len(df) == 2
        assert df["distance"].dtype == dtype

    def test_series_sum(self, ureg: UnitRegistry) -> None:
        dtype = PintDtype("meter", ureg)
        s = pd.Series(PintArray([1.0, 2.0, 3.0], dtype=dtype))
        total = s.sum()
        assert abs(total.magnitude - 6.0) < 1e-10

    def test_series_mean(self, ureg: UnitRegistry) -> None:
        dtype = PintDtype("meter", ureg)
        s = pd.Series(PintArray([1.0, 2.0, 3.0], dtype=dtype))
        avg = s.mean()
        assert abs(avg.magnitude - 2.0) < 1e-10
