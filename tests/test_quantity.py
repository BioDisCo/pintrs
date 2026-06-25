from __future__ import annotations

import pytest
from pintrs import DimensionalityError, Quantity, Unit, UnitRegistry


class TestQuantityCreation:
    def test_from_value_and_units(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.5, "meter")
        assert q.magnitude == 3.5
        assert str(q.units) == "meter"

    def test_from_string(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity("9.8 m/s^2")
        assert abs(q.magnitude - 9.8) < 1e-10

    def test_parse_expression(self, ureg: UnitRegistry) -> None:
        q = ureg.parse_expression("3.5 meters")
        assert abs(q.magnitude - 3.5) < 1e-10

    def test_attribute_access(self, ureg: UnitRegistry) -> None:
        u = ureg.meter
        assert isinstance(u, Unit)
        assert str(u) == "meter"

    def test_symbol_access(self, ureg: UnitRegistry) -> None:
        u = ureg.kg
        assert isinstance(u, Unit)
        assert str(u) == "kilogram"

    def test_dimensionless(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5.0, "dimensionless")
        assert q.dimensionless

    def test_empty_string_raises(self, ureg: UnitRegistry) -> None:
        with pytest.raises(ValueError):
            ureg.Quantity("")


class TestQuantityConversion:
    def test_to(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "kilometer")
        result = q.to("meter")
        assert abs(result.magnitude - 1000.0) < 1e-10

    def test_ito(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "kilometer")
        q.ito("meter")
        assert abs(q.magnitude - 1000.0) < 1e-10

    def test_m_as(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "kilometer")
        assert abs(q.m_as("meter") - 1000.0) < 1e-10

    def test_to_base_units(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "kilometer")
        base = q.to_base_units()
        assert abs(base.magnitude - 1000.0) < 1e-10
        assert str(base.units) == "meter"

    def test_incompatible_raises(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "meter")
        with pytest.raises(ValueError):
            q.to("second")

    def test_prefixed_units(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1.0, "millimeter")
        result = q.to("meter")
        assert abs(result.magnitude - 0.001) < 1e-10

    def test_centimeter_to_inch(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(2.54, "centimeter")
        result = q.to("inch")
        assert abs(result.magnitude - 1.0) < 1e-6


class TestTemperature:
    def test_celsius_to_kelvin(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(100, "degree_Celsius")
        result = q.to("kelvin")
        assert abs(result.magnitude - 373.15) < 1e-6

    def test_celsius_to_fahrenheit(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(100, "degree_Celsius")
        result = q.to("degree_Fahrenheit")
        assert abs(result.magnitude - 212.0) < 1e-6

    def test_fahrenheit_to_celsius(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(32, "degree_Fahrenheit")
        result = q.to("degree_Celsius")
        assert abs(result.magnitude) < 1e-6

    def test_absolute_zero(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(0, "kelvin")
        result = q.to("degree_Celsius")
        assert abs(result.magnitude - (-273.15)) < 1e-6

    def test_celsius_to_base_units_applies_offset(self, ureg: UnitRegistry) -> None:
        # to_base_units must apply the offset, not just the multiplicative factor.
        r = ureg.Quantity(10.0, "degree_Celsius").to_base_units()
        assert str(r.units) == "kelvin"
        assert r.magnitude == pytest.approx(283.15)
        assert ureg.Quantity(100.0, "degree_Celsius").to_base_units().magnitude == (
            pytest.approx(373.15)
        )
        # ito_base_units (in place) too
        q = ureg.Quantity(0.0, "degree_Celsius")
        q.ito_base_units()
        assert q.magnitude == pytest.approx(273.15)

    def test_multiplicative_to_base_units_unchanged(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(2.0, "kilometer").to_base_units().magnitude == (
            pytest.approx(2000.0)
        )


class TestArithmetic:
    def test_add_same_units(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(10, "meter")
        b = ureg.Quantity(5, "meter")
        result = a + b
        assert result.magnitude == 15

    def test_add_compatible_units(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(1, "kilometer")
        b = ureg.Quantity(500, "meter")
        result = a + b
        assert abs(result.magnitude - 1.5) < 1e-10

    def test_add_incompatible_raises(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(1, "meter")
        b = ureg.Quantity(1, "second")
        with pytest.raises(ValueError):
            _ = a + b

    def test_add_number_to_scaled_dimensionless(self, ureg: UnitRegistry) -> None:
        # Issue #5: a dimensionless result carrying intermediate (scaled) units
        # must convert to base units before adding a bare number.
        v1 = ureg.Quantity(4.0, "cm/ms")
        v2 = ureg.Quantity(1.0, "m/s")
        product = v1 / v2  # 40 dimensionless, but stored as cm*s/m/ms
        assert (1 + product).to_base_units().magnitude == pytest.approx(41.0)
        assert (1 + product).dimensionless
        assert (product + 1).to_base_units().magnitude == pytest.approx(41.0)

    def test_sub_number_from_scaled_dimensionless(self, ureg: UnitRegistry) -> None:
        v1 = ureg.Quantity(4.0, "cm/ms")
        v2 = ureg.Quantity(1.0, "m/s")
        product = v1 / v2  # 40 dimensionless
        assert (product - 1).to_base_units().magnitude == pytest.approx(39.0)
        assert (1 - product).to_base_units().magnitude == pytest.approx(-39.0)

    def test_add_number_to_dimensional_still_raises(self, ureg: UnitRegistry) -> None:
        with pytest.raises((ValueError, TypeError)):
            _ = ureg.Quantity(2.0, "meter") + 1

    def _scaled_dimensionless(self, ureg: UnitRegistry) -> Quantity:
        # 40 dimensionless, stored with intermediate units cm*s/m/ms.
        return ureg.Quantity(4.0, "cm/ms") / ureg.Quantity(1.0, "m/s")

    def test_floordiv_number_scaled_dimensionless(self, ureg: UnitRegistry) -> None:
        # Issue #5 (same bug class): // must operate in base units, not raw scale.
        a = self._scaled_dimensionless(ureg)
        assert (a // 7).to_base_units().magnitude == pytest.approx(5.0)  # 40 // 7
        assert (a // 7).dimensionless
        assert (7 // a).to_base_units().magnitude == pytest.approx(0.0)  # 7 // 40

    def test_mod_number_scaled_dimensionless(self, ureg: UnitRegistry) -> None:
        a = self._scaled_dimensionless(ureg)
        assert (a % 7).to_base_units().magnitude == pytest.approx(5.0)  # 40 % 7
        assert (7 % a).to_base_units().magnitude == pytest.approx(7.0)  # 7 % 40
        assert (7 % a).dimensionless

    def test_floordiv_mod_number_dimensional_raises(self, ureg: UnitRegistry) -> None:
        with pytest.raises(ValueError):
            _ = ureg.Quantity(5.0, "meter") // 2
        with pytest.raises(ValueError):
            _ = ureg.Quantity(5.0, "meter") % 2

    def test_ordered_comparison_with_number(self, ureg: UnitRegistry) -> None:
        a = self._scaled_dimensionless(ureg)  # == 40
        assert a < 41
        assert a <= 40
        assert a > 39
        assert a >= 40
        assert not a < 40
        # plain dimensionless quantity vs number
        assert ureg.Quantity(40.0, "") < 41

    def test_ordered_comparison_dimensional_with_number_raises(
        self, ureg: UnitRegistry
    ) -> None:
        with pytest.raises((ValueError, TypeError)):
            _ = ureg.Quantity(5.0, "meter") < 2

    def test_add_collapses_self_rooting_unit_to_dimensionless(
        self, ureg: UnitRegistry
    ) -> None:
        # `radian` is dimensionless but roots to itself; adding a number must
        # collapse to plain dimensionless like pint, not keep `radian`.
        result = ureg.Quantity(5.0, "radian") + 7
        assert result.magnitude == pytest.approx(12.0)
        assert result.dimensionless
        assert str(result.units) in ("", "dimensionless")

    def test_mod_uses_floored_python_semantics(self, ureg: UnitRegistry) -> None:
        # pint/Python modulo: result takes the sign of the divisor.
        a = self._scaled_dimensionless(ureg)  # 40
        assert (a % -3).to_base_units().magnitude == pytest.approx(-2.0)  # 40 % -3
        assert (-3 % a).to_base_units().magnitude == pytest.approx(37.0)  # -3 % 40

    def test_quantity_mod_quantity_floored(self, ureg: UnitRegistry) -> None:
        # quantity % quantity also follows Python floored-modulo sign rules.
        q = ureg.Quantity
        assert (q(-5.0, "m") % q(2.0, "m")).magnitude == pytest.approx(1.0)
        assert (q(5.0, "m") % q(-2.0, "m")).magnitude == pytest.approx(-1.0)
        assert (q(-5.0, "m") // q(2.0, "m")).magnitude == pytest.approx(-3.0)

    def test_floordiv_quantity_converts_units(self, ureg: UnitRegistry) -> None:
        # q // q converts to compatible units and yields a dimensionless count.
        q = ureg.Quantity
        r = q(1.0, "m") // q(3.0, "cm")  # 100 cm // 3 cm
        assert r.magnitude == pytest.approx(33.0)
        assert r.dimensionless
        with pytest.raises(ValueError):
            _ = q(5.0, "m") // q(2.0, "s")  # incompatible dimensions

    def test_divmod(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity
        d, m = divmod(q(5.0, "m"), q(2.0, "m"))
        assert d.magnitude == pytest.approx(2.0)
        assert d.dimensionless
        assert m.magnitude == pytest.approx(1.0)
        assert str(m.units) == "meter"
        # divmod with a bare number on a scaled-dimensionless quantity (issue #5).
        a = self._scaled_dimensionless(ureg)  # 40
        d2, m2 = divmod(a, 7)
        assert d2.to_base_units().magnitude == pytest.approx(5.0)
        assert m2.to_base_units().magnitude == pytest.approx(5.0)

    def test_compare_any_quantity_with_zero(self, ureg: UnitRegistry) -> None:
        # Comparison against 0 is allowed regardless of dimensionality (pint).
        assert not ureg.Quantity(5.0, "meter") < 0
        assert ureg.Quantity(5.0, "meter") >= 0
        assert ureg.Quantity(-5.0, "meter") < 0

    def test_floordiv_dimensional_by_zero_raises_dimensionality(
        self, ureg: UnitRegistry
    ) -> None:
        # Dimensionality is checked before division-by-zero, matching pint.
        with pytest.raises(DimensionalityError):
            _ = ureg.Quantity(5.0, "meter") // 0
        with pytest.raises(DimensionalityError):
            _ = ureg.Quantity(5.0, "meter") % 0

    def test_sub(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(10, "meter")
        b = ureg.Quantity(3, "meter")
        result = a - b
        assert result.magnitude == 7

    def test_mul_quantities(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(10, "meter")
        b = ureg.Quantity(3, "second")
        result = a * b
        assert result.magnitude == 30

    def test_mul_scalar(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5, "meter")
        assert (q * 3).magnitude == 15
        assert (3 * q).magnitude == 15

    def test_div_quantities(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(10, "meter")
        b = ureg.Quantity(2, "second")
        result = a / b
        assert result.magnitude == 5.0

    def test_div_scalar(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(10, "meter")
        assert (q / 2).magnitude == 5.0

    def test_rdiv(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(2, "meter")
        result = 10.0 / q
        assert result.magnitude == 5.0

    def test_pow(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3, "meter")
        result = q**2
        assert result.magnitude == 9

    def test_neg(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(5, "meter")
        assert (-q).magnitude == -5

    def test_abs(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(-5, "meter")
        assert abs(q).magnitude == 5


class TestComparison:
    def test_eq_same(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(10, "meter")
        b = ureg.Quantity(10, "meter")
        assert a == b

    def test_eq_different_units(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(1, "kilometer")
        b = ureg.Quantity(1000, "meter")
        assert a == b

    def test_neq(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(10, "meter")
        b = ureg.Quantity(5, "meter")
        assert a != b

    def test_lt(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(5, "meter") < ureg.Quantity(10, "meter")

    def test_gt(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(10, "meter") > ureg.Quantity(5, "meter")

    def test_le(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(5, "meter") <= ureg.Quantity(5, "meter")
        assert ureg.Quantity(5, "meter") <= ureg.Quantity(10, "meter")

    def test_ge(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(10, "meter") >= ureg.Quantity(10, "meter")
        assert ureg.Quantity(10, "meter") >= ureg.Quantity(5, "meter")


class TestDimensionality:
    def test_length(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "meter")
        assert q.dimensionality == "[length]"
        assert not q.dimensionless

    def test_velocity(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "m/s")
        assert "[length]" in q.dimensionality
        assert "[time]" in q.dimensionality

    def test_dimensionless(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "meter") / ureg.Quantity(1, "meter")
        assert q.dimensionless

    def test_is_compatible_with(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "meter")
        assert q.is_compatible_with("foot")
        assert q.is_compatible_with("kilometer")
        assert not q.is_compatible_with("second")
        assert not q.is_compatible_with("kilogram")

    def test_float_dimensionless(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3, "meter") / ureg.Quantity(1, "meter")
        assert float(q) == 3.0

    def test_float_non_dimensionless_raises(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3, "meter")
        with pytest.raises(ValueError):
            float(q)


class TestRepr:
    def test_repr(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.5, "meter")
        assert "3.5" in repr(q)
        assert "m" in repr(q)

    def test_str(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(3.5, "meter")
        assert str(q) == "3.5 meter"

    def test_hash(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(1, "meter")
        b = ureg.Quantity(1, "meter")
        assert hash(a) == hash(b)


class TestVariousUnits:
    def test_atmosphere_to_pascal(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "atmosphere")
        result = q.to("pascal")
        assert abs(result.magnitude - 101325) < 1

    def test_mile_to_km(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "mile")
        result = q.to("kilometer")
        assert abs(result.magnitude - 1.609344) < 1e-4

    def test_gallon_to_liter(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "gallon")
        result = q.to("liter")
        assert abs(result.magnitude - 3.785412) < 1e-3

    def test_horsepower_to_watt(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "horsepower")
        result = q.to("watt")
        assert abs(result.magnitude - 745.7) < 1.0

    def test_electron_volt(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(1, "electron_volt")
        result = q.to("joule")
        assert abs(result.magnitude - 1.602e-19) < 1e-22

    def test_speed_of_light(self, ureg: UnitRegistry) -> None:
        c = ureg.Quantity(1, "speed_of_light")
        result = c.to("m/s")
        assert abs(result.magnitude - 299792458) < 1


class TestOffsetArithmetic:
    """Offset (temperature) add/subtract follows pint's delta calculus."""

    def test_absolute_plus_absolute_raises(self, ureg: UnitRegistry) -> None:
        from pintrs import OffsetUnitCalculusError

        with pytest.raises(OffsetUnitCalculusError):
            _ = ureg.Quantity(10.0, "degC") + ureg.Quantity(5.0, "degC")
        # adding a plain absolute (kelvin) is also ambiguous
        with pytest.raises(OffsetUnitCalculusError):
            _ = ureg.Quantity(10.0, "degC") + ureg.Quantity(5.0, "kelvin")

    def test_absolute_minus_absolute_is_delta(self, ureg: UnitRegistry) -> None:
        d = ureg.Quantity(10.0, "degC") - ureg.Quantity(5.0, "degC")
        assert d.magnitude == pytest.approx(5.0)
        assert "delta" in str(d.units)

    def test_absolute_plus_delta_stays_absolute(self, ureg: UnitRegistry) -> None:
        r = ureg.Quantity(10.0, "degC") + ureg.Quantity(5.0, "delta_degC")
        assert r.magnitude == pytest.approx(15.0)
        assert str(r.units) == "degree_Celsius"

    def test_absolute_minus_kelvin_is_delta(self, ureg: UnitRegistry) -> None:
        d = ureg.Quantity(10.0, "degC") - ureg.Quantity(5.0, "kelvin")
        assert "delta" in str(d.units)


class TestComplexScalarQuantity:
    """Complex scalar quantities (kept on the complex-capable backend)."""

    def test_abs(self, ureg: UnitRegistry) -> None:
        import numpy as np

        q = ureg.Quantity(3 + 4j, "meter")
        assert complex(np.abs(q).magnitude) == pytest.approx(5.0)
        assert isinstance(q, Quantity)

    def test_conj_preserves_units(self, ureg: UnitRegistry) -> None:
        import numpy as np

        r = np.conj(ureg.Quantity(1 + 2j, "meter"))
        assert complex(r.magnitude) == (1 - 2j)
        assert str(r.units) == "meter"


class TestScalarVsArrayQuantity:
    """Scalar Quantity combined with an array Quantity broadcasts (pint)."""

    def test_eq_lt(self, ureg: UnitRegistry) -> None:
        import numpy as np

        a = ureg.Quantity(np.array([1.0, 2.0]), "meter")
        np.testing.assert_array_equal(ureg.Quantity(1.0, "meter") == a, [True, False])
        np.testing.assert_array_equal(ureg.Quantity(2.0, "meter") < a, [False, False])

    def test_floordiv_mod(self, ureg: UnitRegistry) -> None:
        import numpy as np

        a = ureg.Quantity(np.array([2.0, 3.0]), "meter")
        r = ureg.Quantity(5.0, "meter") // a
        np.testing.assert_allclose([x.magnitude for x in r], [2.0, 1.0])


class TestLogEqualityAndComplexOrdering:
    def test_log_equals_linear(self, ureg: UnitRegistry) -> None:
        assert ureg.Quantity(0.0, "dBW") == ureg.Quantity(1.0, "watt")
        assert ureg.Quantity(0.0, "dBm") == ureg.Quantity(1.0, "milliwatt")

    def test_db_minus_db_is_delta(self, ureg: UnitRegistry) -> None:
        d = ureg.Quantity(3.0, "dB") - ureg.Quantity(1.0, "dB")
        assert d.magnitude == pytest.approx(2.0)
        assert "delta" in str(d.units)

    def test_log_bare_number_arithmetic_uses_linear_value(
        self, ureg: UnitRegistry
    ) -> None:
        # A logarithmic-but-dimensionless unit (dB) combined with a bare number
        # operates on its linear value (1 dB -> 10**0.1), matching pint.
        assert (ureg.Quantity(1.0, "dB") + 5).magnitude == pytest.approx(6.2589254)
        assert (ureg.Quantity(1.0, "dB") - 0.5).magnitude == pytest.approx(0.7589254)
        # `1 dB` linearises to 10**0.1; exact-float equality holds against it.
        assert ureg.Quantity(1.0, "dB") == 10**0.1
        assert ureg.Quantity(2.0, "dB") < 1.7  # 10**0.2 = 1.585 < 1.7

    def test_complex_ordering_raises(self, ureg: UnitRegistry) -> None:
        with pytest.raises(TypeError):
            _ = ureg.Quantity(1 + 2j, "meter") < ureg.Quantity(2 + 0j, "meter")


class TestOffsetUnitCalculus:
    """Multiplying/dividing/powering a non-multiplicative unit is ambiguous and
    raises OffsetUnitCalculusError, matching pint. delta_* units are exempt."""

    @pytest.mark.parametrize("unit", ["degC", "dB", "dBm"])
    def test_mul_by_scalar_raises(self, ureg: UnitRegistry, unit: str) -> None:
        from pintrs import OffsetUnitCalculusError

        with pytest.raises(OffsetUnitCalculusError):
            _ = ureg.Quantity(10.0, unit) * 2
        with pytest.raises(OffsetUnitCalculusError):
            _ = 2 * ureg.Quantity(10.0, unit)

    def test_div_by_scalar_raises(self, ureg: UnitRegistry) -> None:
        from pintrs import OffsetUnitCalculusError

        with pytest.raises(OffsetUnitCalculusError):
            _ = ureg.Quantity(10.0, "degC") / 2
        with pytest.raises(OffsetUnitCalculusError):
            _ = 2 / ureg.Quantity(10.0, "degC")

    def test_mul_two_offset_raises(self, ureg: UnitRegistry) -> None:
        from pintrs import OffsetUnitCalculusError

        with pytest.raises(OffsetUnitCalculusError):
            _ = ureg.Quantity(10.0, "degC") * ureg.Quantity(2.0, "degC")
        with pytest.raises(OffsetUnitCalculusError):
            _ = ureg.Quantity(10.0, "meter") * ureg.Quantity(2.0, "degC")

    def test_pow_raises_except_zero_one(self, ureg: UnitRegistry) -> None:
        from pintrs import OffsetUnitCalculusError

        with pytest.raises(OffsetUnitCalculusError):
            _ = ureg.Quantity(10.0, "degC") ** 2
        assert (ureg.Quantity(10.0, "degC") ** 1).magnitude == 10.0
        assert (ureg.Quantity(10.0, "degC") ** 0).dimensionless

    def test_delta_and_plain_units_unaffected(self, ureg: UnitRegistry) -> None:
        assert (ureg.Quantity(10.0, "delta_degC") * 2).magnitude == 20.0
        assert (ureg.Quantity(3.0, "delta_degC") ** 2).magnitude == 9.0
        assert (ureg.Quantity(10.0, "kelvin") * 2).magnitude == 20.0

    def test_quantity_times_offset_unit_raises(self, ureg: UnitRegistry) -> None:
        from pintrs import OffsetUnitCalculusError

        with pytest.raises(OffsetUnitCalculusError):
            _ = ureg.Quantity(2.0, "meter") * ureg.Unit("degC")
        with pytest.raises(OffsetUnitCalculusError):
            _ = ureg.Quantity(2.0, "meter") / ureg.Unit("degC")

    def test_offset_unit_times_scalar_raises(self, ureg: UnitRegistry) -> None:
        from pintrs import OffsetUnitCalculusError

        with pytest.raises(OffsetUnitCalculusError):
            _ = ureg.Unit("degC") * 2
        with pytest.raises(OffsetUnitCalculusError):
            _ = 2 * ureg.Unit("degC")
        # Composing two units (no magnitude) is allowed.
        assert "degree_Celsius" in str(ureg.Unit("degC") * ureg.Unit("meter"))

    def test_array_offset_arithmetic_raises(self, ureg: UnitRegistry) -> None:
        import numpy as np
        from pintrs import OffsetUnitCalculusError

        arr = ureg.Quantity(np.array([1.0, 2.0]), "degC")
        with pytest.raises(OffsetUnitCalculusError):
            _ = arr * 2
        with pytest.raises(OffsetUnitCalculusError):
            _ = 2 * arr
        with pytest.raises(OffsetUnitCalculusError):
            _ = arr / 2
        with pytest.raises(OffsetUnitCalculusError):
            _ = arr**2
        with pytest.raises(OffsetUnitCalculusError):
            _ = ureg.Quantity(np.array([1.0, 2.0]), "meter") * arr

    def test_array_ordinary_arithmetic_unaffected(self, ureg: UnitRegistry) -> None:
        import numpy as np

        arr = ureg.Quantity(np.array([1.0, 2.0]), "km")
        np.testing.assert_allclose((arr * 2).magnitude, [2.0, 4.0])
        np.testing.assert_allclose((arr / 2).magnitude, [0.5, 1.0])
        np.testing.assert_allclose((arr**2).magnitude, [1.0, 4.0])


class TestComplexOffsetLogConversion:
    """Complex magnitudes keep their imaginary part through offset/log
    conversions, applying the affine/exponential/logarithmic transform that
    pint applies element-wise (regression: a real-only cast truncated them)."""

    def test_complex_offset_affine(self, ureg: UnitRegistry) -> None:
        r = ureg.Quantity(1 + 2j, "degC").to("kelvin")
        assert r.magnitude == pytest.approx(274.15 + 2j)

    def test_complex_from_log_exponential(self, ureg: UnitRegistry) -> None:
        r = ureg.Quantity(1 + 2j, "dBW").to("watt")
        assert r.magnitude == pytest.approx(1.1277741482491686 + 0.5594807083376298j)

    def test_complex_to_log_logarithmic(self, ureg: UnitRegistry) -> None:
        r = ureg.Quantity(1 + 2j, "watt").to("dBW")
        assert r.magnitude == pytest.approx(3.4948500216800933 + 4.808285787842339j)

    def test_complex_array_offset(self, ureg: UnitRegistry) -> None:
        import numpy as np

        r = ureg.Quantity(np.array([1 + 2j, 3 + 0j]), "degC").to("kelvin")
        np.testing.assert_allclose(r.magnitude, [274.15 + 2j, 276.15 + 0j])

    def test_complex_offset_to_offset_affine(self, ureg: UnitRegistry) -> None:
        # degC -> degF is affine; classification must not treat it as log.
        r = ureg.Quantity(1 + 2j, "degC").to("degF")
        assert r.magnitude == pytest.approx(33.8 + 3.6j)

    def test_complex_log_to_log_affine(self, ureg: UnitRegistry) -> None:
        # dBW -> dBm differ by a constant offset (affine in dB space).
        r = ureg.Quantity(1 + 2j, "dBW").to("dBm")
        assert r.magnitude == pytest.approx(31 + 2j)
