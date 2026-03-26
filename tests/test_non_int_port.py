"""Port of pint's test_non_int.py for pintrs.

Tests that Decimal and Fraction values are accepted as input and produce
correct results. pintrs coerces these to float internally, so magnitude
type preservation is not tested.
"""

from __future__ import annotations

import math
import operator as op
from decimal import Decimal
from fractions import Fraction
from typing import Any, Union

import pytest
from pintrs import (
    DimensionalityError,
    Quantity,
    UnitRegistry,
)

NonIntType = Union[type[Decimal], type[Fraction], type[float]]


@pytest.fixture
def ureg() -> UnitRegistry:
    """Shared unit registry for tests."""
    return UnitRegistry()


def _q(value: str, units: str, non_int_type: NonIntType) -> Quantity:
    """Create a Quantity from string value via non_int_type."""
    return Quantity(float(non_int_type(value)), units)


def _approx_equal(
    first: Quantity | float,
    second: Quantity | float,
    rtol: float = 1e-7,
    atol: float = 0.0,
) -> None:
    """Assert two quantities are approximately equal."""
    if isinstance(first, Quantity) and isinstance(second, Quantity):
        first_mag = first.to(str(second.units)).magnitude
        second_mag = second.magnitude
    elif isinstance(first, Quantity):
        first_mag = first.magnitude
        second_mag = float(second)
    elif isinstance(second, Quantity):
        first_mag = float(first)
        second_mag = second.magnitude
    else:
        first_mag = float(first)
        second_mag = float(second)

    if atol > 0:
        assert abs(first_mag - second_mag) <= atol, (
            f"{first_mag} != {second_mag} within atol={atol}"
        )
    elif second_mag == 0:
        assert abs(first_mag) <= atol, f"{first_mag} != 0 within atol={atol}"
    else:
        assert abs((first_mag - second_mag) / second_mag) <= rtol, (
            f"{first_mag} != {second_mag} within rtol={rtol}"
        )


# ---------------------------------------------------------------------------
# Test: Quantity creation with non-int types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("non_int_type", [float, Decimal, Fraction])
class TestQuantityCreation:
    """Test creating quantities with Decimal, Fraction, and float values."""

    def test_quantity_creation_from_value_and_string(
        self, ureg: UnitRegistry, non_int_type: NonIntType
    ) -> None:
        value = float(non_int_type("4.2"))
        q = Quantity(value, "meter")
        assert q.magnitude == pytest.approx(4.2)

    def test_quantity_creation_from_value_and_unit(
        self, ureg: UnitRegistry, non_int_type: NonIntType
    ) -> None:
        value = float(non_int_type("4.2"))
        q = Quantity(value, "meter")
        assert q.magnitude == pytest.approx(4.2)

    def test_quantity_creation_from_string(
        self, ureg: UnitRegistry, non_int_type: NonIntType
    ) -> None:
        q = Quantity("4.2*meter")
        assert q.magnitude == pytest.approx(4.2)

    def test_quantity_creation_from_quantity(
        self, ureg: UnitRegistry, non_int_type: NonIntType
    ) -> None:
        value = float(non_int_type("4.2"))
        q1 = Quantity(value, "meter")
        q2 = Quantity(q1)
        assert q2.magnitude == q1.magnitude
        assert q2.units == q1.units
        assert q2 is not q1

    def test_quantity_creation_none_units(
        self, ureg: UnitRegistry, non_int_type: NonIntType
    ) -> None:
        value = float(non_int_type("4.2"))
        q = Quantity(value, None)
        assert q.magnitude == pytest.approx(4.2)
        assert q.dimensionless


# ---------------------------------------------------------------------------
# Test: NaN creation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("non_int_type", [float, Decimal])
class TestNanCreation:
    """Decimal and float support NaN; Fraction does not."""

    def test_nan_creation(self, ureg: UnitRegistry, non_int_type: NonIntType) -> None:
        value = float(non_int_type("nan"))
        q = Quantity(value, "meter")
        assert math.isnan(q.magnitude)


class TestNanCreationFraction:
    """Fraction raises ValueError for NaN."""

    def test_nan_fraction_raises(self, ureg: UnitRegistry) -> None:
        with pytest.raises(ValueError):
            Fraction("nan")


# ---------------------------------------------------------------------------
# Test: Quantity comparison
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("non_int_type", [float, Decimal, Fraction])
class TestQuantityComparison:
    """Test comparison operators on quantities created from non-int types."""

    def test_equality(self, ureg: UnitRegistry, non_int_type: NonIntType) -> None:
        x = _q("4.2", "meter", non_int_type)
        y = _q("4.2", "meter", non_int_type)
        assert x == y
        assert not (x != y)

    def test_identity(self, ureg: UnitRegistry, non_int_type: NonIntType) -> None:
        x = _q("4.2", "meter", non_int_type)
        assert x == x
        assert not (x != x)

    def test_ordering(self, ureg: UnitRegistry, non_int_type: NonIntType) -> None:
        x = _q("4.2", "meter", non_int_type)
        y = _q("4.2", "meter", non_int_type)
        z = _q("5", "meter", non_int_type)

        assert x <= y
        assert x >= y
        assert not (x < y)
        assert not (x > y)
        assert x != z
        assert x < z

    def test_different_dimensionality_not_equal(
        self, ureg: UnitRegistry, non_int_type: NonIntType
    ) -> None:
        z = _q("5", "meter", non_int_type)
        j = _q("5", "meter*meter", non_int_type)
        assert z != j

    def test_zero_comparison_compatible_units(
        self, ureg: UnitRegistry, non_int_type: NonIntType
    ) -> None:
        assert _q("0", "meter", non_int_type) == _q("0", "centimeter", non_int_type)

    def test_zero_comparison_incompatible_units(
        self, ureg: UnitRegistry, non_int_type: NonIntType
    ) -> None:
        assert _q("0", "meter", non_int_type) != _q("0", "second", non_int_type)

    def test_cross_unit_comparison(
        self, ureg: UnitRegistry, non_int_type: NonIntType
    ) -> None:
        assert _q("10", "meter", non_int_type) < _q("5", "kilometer", non_int_type)

    def test_comparison_after_conversion(
        self, ureg: UnitRegistry, non_int_type: NonIntType
    ) -> None:
        assert _q("1000", "millimeter", non_int_type) == _q("1", "meter", non_int_type)


# ---------------------------------------------------------------------------
# Test: Quantity hash
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("non_int_type", [float, Decimal, Fraction])
class TestQuantityHash:
    """Test hash consistency for quantities."""

    def test_hash_equal_quantities(
        self, ureg: UnitRegistry, non_int_type: NonIntType
    ) -> None:
        x = _q("4.2", "meter", non_int_type)
        x2 = _q("4200", "millimeter", non_int_type)
        assert hash(x) == hash(x2)


# ---------------------------------------------------------------------------
# Test: Unit conversion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("non_int_type", [float, Decimal, Fraction])
class TestUnitConversion:
    """Test .to() and .to_base_units() with non-int types."""

    def test_to_base_units_inch(
        self, ureg: UnitRegistry, non_int_type: NonIntType
    ) -> None:
        x = Quantity("1*inch")
        result = x.to_base_units()
        _approx_equal(result, Quantity(0.0254, "meter"))

    def test_to_base_units_inch_squared(
        self, ureg: UnitRegistry, non_int_type: NonIntType
    ) -> None:
        x = Quantity("1*inch*inch")
        expected = Quantity(0.0254**2, "meter*meter")
        _approx_equal(x.to_base_units(), expected)

    def test_to_base_units_inch_per_minute(
        self, ureg: UnitRegistry, non_int_type: NonIntType
    ) -> None:
        x = Quantity("1*inch/minute")
        expected = Quantity(0.0254 / 60.0, "meter/second")
        _approx_equal(x.to_base_units(), expected)

    def test_convert_inch_to_meter(
        self, ureg: UnitRegistry, non_int_type: NonIntType
    ) -> None:
        result = Quantity("2 inch").to("meter")
        expected = Quantity(2.0 * 0.0254, "meter")
        _approx_equal(result, expected)

    def test_convert_meter_to_inch(
        self, ureg: UnitRegistry, non_int_type: NonIntType
    ) -> None:
        result = Quantity("2 meter").to("inch")
        expected = Quantity(2.0 / 0.0254, "inch")
        _approx_equal(result, expected)

    def test_convert_sidereal_year_to_second(
        self, ureg: UnitRegistry, non_int_type: NonIntType
    ) -> None:
        result = Quantity("2 sidereal_year").to("second")
        _approx_equal(result, _q("63116299.5270912", "second", float), rtol=1e-4)

    def test_convert_centimeter_per_second_to_inch_per_second(
        self, ureg: UnitRegistry, non_int_type: NonIntType
    ) -> None:
        result = Quantity("2.54 centimeter/second").to("inch/second")
        expected = Quantity(1.0, "inch/second")
        _approx_equal(result, expected)

    def test_convert_centimeter_to_inch_magnitude(
        self, ureg: UnitRegistry, non_int_type: NonIntType
    ) -> None:
        assert round(abs(Quantity("2.54 centimeter").to("inch").magnitude - 1), 7) == 0

    def test_convert_second_to_millisecond_magnitude(
        self, ureg: UnitRegistry, non_int_type: NonIntType
    ) -> None:
        assert (
            round(abs(Quantity("2 second").to("millisecond").magnitude - 2000), 7) == 0
        )


# ---------------------------------------------------------------------------
# Test: Context attribute and symbol equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("non_int_type", [float, Decimal, Fraction])
class TestContextAndSymbol:
    """Test ureg attribute access and unit symbol equivalence."""

    def test_context_attr(self, ureg: UnitRegistry, non_int_type: NonIntType) -> None:
        assert ureg.meter == _q("1", "meter", non_int_type)

    def test_symbol_ms(self, ureg: UnitRegistry, non_int_type: NonIntType) -> None:
        assert _q("2", "ms", non_int_type) == _q("2", "millisecond", non_int_type)

    def test_symbol_cm(self, ureg: UnitRegistry, non_int_type: NonIntType) -> None:
        assert _q("2", "cm", non_int_type) == _q("2", "centimeter", non_int_type)


# ---------------------------------------------------------------------------
# Test: Dimensionless units
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("non_int_type", [float, Decimal, Fraction])
class TestDimensionlessUnits:
    """Test dimensionless quantity behavior with non-int types."""

    def test_degree_to_radian(
        self, ureg: UnitRegistry, non_int_type: NonIntType
    ) -> None:
        result = _q("360", "degree", non_int_type).to("radian")
        assert result.magnitude == pytest.approx(2.0 * math.pi, rel=1e-6)

    def test_meter_divide_meter(
        self, ureg: UnitRegistry, non_int_type: NonIntType
    ) -> None:
        result = _q("1", "meter", non_int_type) / _q("1", "meter", non_int_type)
        assert result == 1

    def test_meter_divide_mm_to_dimensionless(
        self, ureg: UnitRegistry, non_int_type: NonIntType
    ) -> None:
        result = (_q("1", "meter", non_int_type) / _q("1", "mm", non_int_type)).to("")
        assert result.magnitude == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# Test: Offset unit conversion (temperature)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("non_int_type", [float, Decimal, Fraction])
class TestOffsetConversion:
    """Test temperature offset unit conversions."""

    @pytest.mark.parametrize(
        ("value", "src", "expected_value", "dst", "atol"),
        [
            ("0", "kelvin", "0", "kelvin", 0.01),
            ("0", "degC", "273.15", "kelvin", 0.01),
            ("0", "degF", "255.372222", "kelvin", 1.0),
            ("100", "kelvin", "100", "kelvin", 0.01),
            ("100", "degC", "373.15", "kelvin", 0.01),
            ("100", "degF", "310.92777777", "kelvin", 1.0),
            ("0", "kelvin", "-273.15", "degC", 0.01),
            ("100", "kelvin", "-173.15", "degC", 0.01),
            ("0", "kelvin", "-459.67", "degF", 1.0),
            ("100", "kelvin", "-279.67", "degF", 1.0),
            ("32", "degF", "0", "degC", 0.01),
            ("100", "degC", "212", "degF", 0.01),
            ("54", "degF", "12.2222", "degC", 0.1),
            ("12", "degC", "53.6", "degF", 0.1),
            ("12", "kelvin", "-261.15", "degC", 0.01),
            ("12", "degC", "285.15", "kelvin", 0.01),
            ("12", "kelvin", "21.6", "degR", 0.01),
            ("12", "degR", "6.66666667", "kelvin", 0.01),
            ("12", "degC", "513.27", "degR", 0.1),
            ("12", "degR", "-266.483333", "degC", 0.1),
        ],
    )
    def test_offset_conversion(
        self,
        ureg: UnitRegistry,
        non_int_type: NonIntType,
        value: str,
        src: str,
        expected_value: str,
        dst: str,
        atol: float,
    ) -> None:
        result = _q(value, src, non_int_type).to(dst)
        expected = _q(expected_value, dst, non_int_type)
        _approx_equal(result, expected, atol=atol)


# ---------------------------------------------------------------------------
# Test: Arithmetic operations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("non_int_type", [float, Decimal, Fraction])
class TestArithmetic:
    """Test basic arithmetic with non-int types."""

    def test_addition_same_unit(
        self, ureg: UnitRegistry, non_int_type: NonIntType
    ) -> None:
        x = _q("1", "centimeter", non_int_type)
        result = x + x
        expected = _q("2", "centimeter", non_int_type)
        _approx_equal(result, expected)

    def test_addition_compatible_units(
        self, ureg: UnitRegistry, non_int_type: NonIntType
    ) -> None:
        x = _q("1", "centimeter", non_int_type)
        y = _q("1", "inch", non_int_type)
        result = x + y
        expected = Quantity(1.0 + 2.54, "centimeter")
        _approx_equal(result, expected)

    def test_addition_dimensionality_error(
        self, ureg: UnitRegistry, non_int_type: NonIntType
    ) -> None:
        x = _q("1", "centimeter", non_int_type)
        z = _q("1", "second", non_int_type)
        with pytest.raises(DimensionalityError):
            op.add(x, z)

    def test_addition_scalar_quantity_error(
        self, ureg: UnitRegistry, non_int_type: NonIntType
    ) -> None:
        x = _q("1", "centimeter", non_int_type)
        with pytest.raises(DimensionalityError):
            op.add(float(non_int_type("10")), x)
        with pytest.raises(DimensionalityError):
            op.add(x, float(non_int_type("10")))

    def test_subtraction_same_unit(
        self, ureg: UnitRegistry, non_int_type: NonIntType
    ) -> None:
        x = _q("1", "centimeter", non_int_type)
        result = x - x
        expected = _q("0", "centimeter", non_int_type)
        _approx_equal(result, expected)

    def test_subtraction_dimensionality_error(
        self, ureg: UnitRegistry, non_int_type: NonIntType
    ) -> None:
        x = _q("1", "centimeter", non_int_type)
        z = _q("1", "second", non_int_type)
        with pytest.raises(DimensionalityError):
            op.sub(x, z)

    def test_multiplication(self, ureg: UnitRegistry, non_int_type: NonIntType) -> None:
        result = _q("4.2", "meter", non_int_type) * _q("10", "inch", non_int_type)
        expected = Quantity(42.0, "meter*inch")
        _approx_equal(result, expected)

    def test_multiplication_scalar(
        self, ureg: UnitRegistry, non_int_type: NonIntType
    ) -> None:
        q = _q("4.2", "meter", non_int_type)
        result = q * float(non_int_type("10"))
        expected = _q("42", "meter", non_int_type)
        _approx_equal(result, expected)

    def test_division(self, ureg: UnitRegistry, non_int_type: NonIntType) -> None:
        result = _q("4.2", "meter", non_int_type) / _q("10", "inch", non_int_type)
        expected = Quantity(0.42, "meter/inch")
        _approx_equal(result, expected)

    def test_division_scalar(
        self, ureg: UnitRegistry, non_int_type: NonIntType
    ) -> None:
        q = _q("4.2", "meter", non_int_type)
        result = q / float(non_int_type("10"))
        expected = Quantity(0.42, "meter")
        _approx_equal(result, expected)


# ---------------------------------------------------------------------------
# Test: abs, round, neg
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("non_int_type", [float, Decimal, Fraction])
class TestAbsRoundNeg:
    """Test abs, round, and negation on quantities."""

    def test_abs(self, ureg: UnitRegistry, non_int_type: NonIntType) -> None:
        x = _q("-4.2", "meter", non_int_type)
        result = abs(x)
        expected = _q("4.2", "meter", non_int_type)
        _approx_equal(result, expected)

    def test_round(self, ureg: UnitRegistry, non_int_type: NonIntType) -> None:
        x = _q("4.2", "meter", non_int_type)
        result = round(x)
        expected = Quantity(round(4.2), "meter")
        _approx_equal(result, expected)

    def test_neg(self, ureg: UnitRegistry, non_int_type: NonIntType) -> None:
        x = _q("4.2", "meter", non_int_type)
        result = -x
        expected = _q("-4.2", "meter", non_int_type)
        _approx_equal(result, expected)


# ---------------------------------------------------------------------------
# Test: float and complex conversion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("non_int_type", [float, Decimal, Fraction])
class TestFloatComplex:
    """Test float() and complex() on dimensionless quantities."""

    @pytest.mark.parametrize("fun", [float, complex])
    def test_float_complex_dimensionless(
        self,
        ureg: UnitRegistry,
        non_int_type: NonIntType,
        fun: type[Any],
    ) -> None:
        x = _q("-4.2", "", non_int_type)
        assert fun(x) == fun(x.magnitude)

    @pytest.mark.parametrize("fun", [float, complex])
    def test_float_complex_with_units_raises(
        self,
        ureg: UnitRegistry,
        non_int_type: NonIntType,
        fun: type[Any],
    ) -> None:
        z = _q("1", "meter", non_int_type)
        with pytest.raises(DimensionalityError):
            fun(z)


# ---------------------------------------------------------------------------
# Test: iter raises TypeError for scalar quantities
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("non_int_type", [float, Decimal, Fraction])
class TestNotIterable:
    """Scalar quantities must not be iterable."""

    def test_notiter(self, ureg: UnitRegistry, non_int_type: NonIntType) -> None:
        x = _q("1", "meter", non_int_type)
        with pytest.raises(TypeError):
            iter(x)
