"""Tests for numpy array quantity support."""

from __future__ import annotations

import pytest

numpy = pytest.importorskip("numpy")

from pintrs import UnitRegistry
from pintrs.numpy_support import ArrayQuantity


@pytest.fixture
def ureg() -> UnitRegistry:
    return UnitRegistry()


class TestArrayCreation:
    def test_from_array(self, ureg: UnitRegistry) -> None:
        q = ArrayQuantity(numpy.array([1.0, 2.0, 3.0]), "meter", ureg)
        assert q.shape == (3,)
        assert q.ndim == 1
        numpy.testing.assert_array_equal(q.magnitude, [1.0, 2.0, 3.0])

    def test_from_list(self, ureg: UnitRegistry) -> None:
        q = ArrayQuantity([1.0, 2.0], "meter", ureg)
        assert q.shape == (2,)

    def test_2d(self, ureg: UnitRegistry) -> None:
        q = ArrayQuantity(numpy.array([[1, 2], [3, 4]]), "meter", ureg)
        assert q.shape == (2, 2)
        assert q.ndim == 2


class TestArrayConversion:
    def test_to(self, ureg: UnitRegistry) -> None:
        q = ArrayQuantity(numpy.array([1.0, 2.0]), "kilometer", ureg)
        result = q.to("meter")
        numpy.testing.assert_array_almost_equal(result.magnitude, [1000.0, 2000.0])

    def test_m_as(self, ureg: UnitRegistry) -> None:
        q = ArrayQuantity(numpy.array([1.0, 2.0]), "kilometer", ureg)
        result = q.m_as("meter")
        numpy.testing.assert_array_almost_equal(result, [1000.0, 2000.0])

    def test_to_base_units(self, ureg: UnitRegistry) -> None:
        q = ArrayQuantity(numpy.array([1.0, 2.0]), "kilometer", ureg)
        result = q.to_base_units()
        numpy.testing.assert_array_almost_equal(result.magnitude, [1000.0, 2000.0])

    def test_to_compact(self, ureg: UnitRegistry) -> None:
        q = ArrayQuantity(numpy.array([1500, 2500]), "meter", ureg)
        result = q.to_compact()
        numpy.testing.assert_array_almost_equal(result.magnitude, [1.5, 2.5])


class TestArrayArithmetic:
    def test_add_same_units(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(numpy.array([1.0, 2.0]), "meter", ureg)
        b = ArrayQuantity(numpy.array([3.0, 4.0]), "meter", ureg)
        result = a + b
        numpy.testing.assert_array_equal(result.magnitude, [4.0, 6.0])

    def test_add_compatible_units(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(numpy.array([1.0, 2.0]), "kilometer", ureg)
        b = ArrayQuantity(numpy.array([500.0, 1000.0]), "meter", ureg)
        result = a + b
        numpy.testing.assert_array_almost_equal(result.magnitude, [1.5, 3.0])

    def test_sub(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(numpy.array([10.0, 20.0]), "meter", ureg)
        b = ArrayQuantity(numpy.array([3.0, 5.0]), "meter", ureg)
        result = a - b
        numpy.testing.assert_array_equal(result.magnitude, [7.0, 15.0])

    def test_mul_scalar(self, ureg: UnitRegistry) -> None:
        q = ArrayQuantity(numpy.array([1.0, 2.0]), "meter", ureg)
        result = q * 3
        numpy.testing.assert_array_equal(result.magnitude, [3.0, 6.0])

    def test_mul_quantity(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(numpy.array([2.0, 3.0]), "meter", ureg)
        b = ArrayQuantity(numpy.array([4.0, 5.0]), "second", ureg)
        result = a * b
        numpy.testing.assert_array_equal(result.magnitude, [8.0, 15.0])

    def test_div(self, ureg: UnitRegistry) -> None:
        q = ArrayQuantity(numpy.array([10.0, 20.0]), "meter", ureg)
        result = q / 2
        numpy.testing.assert_array_equal(result.magnitude, [5.0, 10.0])

    def test_pow(self, ureg: UnitRegistry) -> None:
        q = ArrayQuantity(numpy.array([2.0, 3.0]), "meter", ureg)
        result = q**2
        numpy.testing.assert_array_equal(result.magnitude, [4.0, 9.0])

    def test_neg(self, ureg: UnitRegistry) -> None:
        q = ArrayQuantity(numpy.array([1.0, -2.0]), "meter", ureg)
        result = -q
        numpy.testing.assert_array_equal(result.magnitude, [-1.0, 2.0])

    def test_abs(self, ureg: UnitRegistry) -> None:
        q = ArrayQuantity(numpy.array([-1.0, 2.0]), "meter", ureg)
        result = abs(q)
        numpy.testing.assert_array_equal(result.magnitude, [1.0, 2.0])


class TestArrayComparison:
    def test_gt(self, ureg: UnitRegistry) -> None:
        q = ArrayQuantity(numpy.array([1.0, 2.0, 3.0]), "meter", ureg)
        result = q > ArrayQuantity(numpy.array([1.5, 1.5, 1.5]), "meter", ureg)
        numpy.testing.assert_array_equal(result, [False, True, True])

    def test_eq(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(numpy.array([1.0, 2.0]), "meter", ureg)
        b = ArrayQuantity(numpy.array([1.0, 2.0]), "meter", ureg)
        numpy.testing.assert_array_equal(a == b, [True, True])


class TestArrayIndexing:
    def test_getitem_scalar(self, ureg: UnitRegistry) -> None:
        q = ArrayQuantity(numpy.array([1.0, 2.0, 3.0]), "meter", ureg)
        assert q[0].magnitude == 1.0

    def test_getitem_slice(self, ureg: UnitRegistry) -> None:
        q = ArrayQuantity(numpy.array([1.0, 2.0, 3.0]), "meter", ureg)
        result = q[1:]
        assert isinstance(result, ArrayQuantity)
        numpy.testing.assert_array_equal(result.magnitude, [2.0, 3.0])


class TestArrayProperties:
    def test_dimensionality(self, ureg: UnitRegistry) -> None:
        q = ArrayQuantity(numpy.array([1.0]), "meter", ureg)
        assert q.dimensionality == "[length]"

    def test_dimensionless(self, ureg: UnitRegistry) -> None:
        q = ArrayQuantity(numpy.array([1.0]), "dimensionless", ureg)
        assert q.dimensionless
        assert q.unitless

    def test_transpose(self, ureg: UnitRegistry) -> None:
        q = ArrayQuantity(numpy.array([[1, 2], [3, 4]]), "meter", ureg)
        result = q.T
        assert result.shape == (2, 2)
        assert result.magnitude[0][1] == 3

    def test_dtype(self, ureg: UnitRegistry) -> None:
        q = ArrayQuantity(numpy.array([1.0, 2.0]), "meter", ureg)
        assert q.dtype == numpy.float64


class TestArrayNumpy:
    def test_sqrt(self, ureg: UnitRegistry) -> None:
        q = ArrayQuantity(numpy.array([4.0, 9.0]), "m ** 2", ureg)
        result = numpy.sqrt(q)
        numpy.testing.assert_array_almost_equal(result.magnitude, [2.0, 3.0])

    def test_multiply(self, ureg: UnitRegistry) -> None:
        a = ArrayQuantity(numpy.array([2.0, 3.0]), "meter", ureg)
        result = numpy.multiply(a, 2)
        numpy.testing.assert_array_equal(result.magnitude, [4.0, 6.0])

    def test_array_protocol(self, ureg: UnitRegistry) -> None:
        q = ArrayQuantity(numpy.array([1.0, 2.0]), "meter", ureg)
        arr = numpy.asarray(q)
        numpy.testing.assert_array_equal(arr, [1.0, 2.0])


class TestScalarTranscendental:
    """Regression tests for GH #4: ufuncs on dimensionless scalar quantities."""

    @pytest.mark.parametrize(
        ("func", "expected"),
        [
            (numpy.sin, numpy.sin(1.0)),
            (numpy.cos, numpy.cos(1.0)),
            (numpy.exp, numpy.exp(1.0)),
            (numpy.log, numpy.log(1.0)),
        ],
    )
    def test_dimensionless_scalar_ufunc(
        self,
        ureg: UnitRegistry,
        func: object,
        expected: float,
    ) -> None:
        q = ureg.Quantity(10, "s") / ureg.Quantity(10, "s")
        result = func(q)  # type: ignore[operator]
        assert result.dimensionless
        assert result.magnitude == pytest.approx(expected)

    def test_scalar_trig_radians(self, ureg: UnitRegistry) -> None:
        result = numpy.sin(ureg.Quantity(numpy.pi / 2, "rad"))
        assert result.magnitude == pytest.approx(1.0)

    def test_scalar_trig_degrees(self, ureg: UnitRegistry) -> None:
        result = numpy.sin(ureg.Quantity(90, "deg"))
        assert result.magnitude == pytest.approx(1.0)

    def test_scalar_trig_non_dimensionless_raises(
        self,
        ureg: UnitRegistry,
    ) -> None:
        from pintrs import DimensionalityError

        with pytest.raises(DimensionalityError):
            numpy.sin(ureg.Quantity(10, "s"))

    @pytest.mark.parametrize(
        ("func", "ratio", "expected"),
        [
            # 8 ms / 8 s is dimensionless but scales to 0.001, so the conversion
            # factor must be folded in before the ufunc (GH #4 follow-up).
            (numpy.sin, "*", numpy.sin(numpy.pi * 0.001)),
            (numpy.cos, "*", numpy.cos(numpy.pi * 0.001)),
            (numpy.exp, "", numpy.exp(0.001)),
            (numpy.arcsin, "", numpy.arcsin(0.001)),
        ],
    )
    def test_scaled_dimensionless_applies_conversion_factor(
        self,
        ureg: UnitRegistry,
        func: object,
        ratio: str,
        expected: float,
    ) -> None:
        q = ureg.Quantity(8, "ms") / ureg.Quantity(8, "s")
        arg = numpy.pi * q if ratio == "*" else q
        result = func(arg)  # type: ignore[operator]
        assert result.magnitude == pytest.approx(expected)

    def test_scaled_dimensionless_log(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(8, "s") / ureg.Quantity(8, "ms")
        result = numpy.log(q)
        assert result.magnitude == pytest.approx(numpy.log(1000.0))

    def test_scaled_dimensionless_array(self, ureg: UnitRegistry) -> None:
        q = ureg.Quantity(numpy.array([8.0, 16.0]), "ms") / ureg.Quantity(
            numpy.array([8.0, 8.0]),
            "s",
        )
        result = numpy.sin(numpy.pi * q)
        numpy.testing.assert_allclose(
            result.magnitude,
            numpy.sin(numpy.pi * numpy.array([0.001, 0.002])),
        )


class TestBinaryUfuncScaling:
    """Binary ufuncs must scale every dimensionless operand by its own factor."""

    def test_logaddexp_both_scaled_operands(self, ureg: UnitRegistry) -> None:
        r = ureg.Quantity(8, "ms") / ureg.Quantity(8, "s")
        result = numpy.logaddexp(r, r)
        assert result.magnitude == pytest.approx(numpy.logaddexp(0.001, 0.001))

    def test_logaddexp_quantity_at_index_one(self, ureg: UnitRegistry) -> None:
        r = ureg.Quantity(1000, "ms") / ureg.Quantity(1, "s")
        result = numpy.logaddexp(ureg.Quantity(0, "dimensionless"), r)
        assert result.magnitude == pytest.approx(numpy.logaddexp(0.0, 1.0))

    def test_logaddexp_rejects_dimensional_operand(self, ureg: UnitRegistry) -> None:
        from pintrs import DimensionalityError

        with pytest.raises(DimensionalityError):
            numpy.logaddexp(ureg.Quantity(0, "dimensionless"), ureg.Quantity(1, "m"))

    def test_arctan2_quantity_with_bare_number(self, ureg: UnitRegistry) -> None:
        r = ureg.Quantity(1, "ms") / ureg.Quantity(1, "s")
        assert numpy.arctan2(r, 1).magnitude == pytest.approx(numpy.arctan2(0.001, 1.0))
        assert numpy.arctan2(1, r).magnitude == pytest.approx(numpy.arctan2(1.0, 0.001))

    def test_arctan2_rejects_dimensional_with_bare_number(
        self,
        ureg: UnitRegistry,
    ) -> None:
        from pintrs import DimensionalityError

        with pytest.raises(DimensionalityError):
            numpy.arctan2(ureg.Quantity(1, "m"), 1)

    def test_power_scaled_dimensionless_exponent(self, ureg: UnitRegistry) -> None:
        exp = ureg.Quantity(1000, "ms") / ureg.Quantity(1, "s")
        result = numpy.power(ureg.Quantity(2, "m"), exp)
        assert result.magnitude == pytest.approx(2.0)
        assert result.dimensionality == ureg.Quantity(1, "m").dimensionality

    def test_power_rejects_dimensional_exponent(self, ureg: UnitRegistry) -> None:
        from pintrs import DimensionalityError

        with pytest.raises(DimensionalityError):
            numpy.power(ureg.Quantity(2, "m"), ureg.Quantity(2, "m"))


class TestBareNumberWithUnits:
    """Matching-dims ufuncs mixing a quantity with a bare number (pint parity)."""

    DIMENSIONAL_RAISERS = [
        (numpy.add, lambda q: (q(1, "m"), 5)),
        (numpy.subtract, lambda q: (q(1, "m"), 2)),
        (numpy.add, lambda q: (5, q(1, "m"))),
        (numpy.maximum, lambda q: (q(1, "m"), 2)),
        (numpy.minimum, lambda q: (q(1, "m"), 2)),
        (numpy.hypot, lambda q: (q(3, "m"), 4)),
        (numpy.copysign, lambda q: (q(5, "m"), -2)),
        (numpy.nextafter, lambda q: (q(1, "m"), 2)),
        (numpy.less, lambda q: (q(3, "m"), 4)),
        (numpy.greater, lambda q: (q(3, "m"), 4)),
        (numpy.equal, lambda q: (q(3, "m"), 4)),
        (numpy.not_equal, lambda q: (q(3, "m"), 4)),
    ]

    @pytest.mark.parametrize(("func", "args"), DIMENSIONAL_RAISERS)
    def test_dimensional_quantity_with_bare_number_raises(
        self,
        ureg: UnitRegistry,
        func: object,
        args: object,
    ) -> None:
        from pintrs import DimensionalityError

        a, b = args(ureg.Quantity)  # type: ignore[operator]
        with pytest.raises(DimensionalityError):
            func(a, b)  # type: ignore[operator]

    def test_scaled_dimensionless_bare_number_converted(
        self,
        ureg: UnitRegistry,
    ) -> None:
        r = ureg.Quantity(1000, "ms") / ureg.Quantity(1, "s")
        # 1 (plain dimensionless) == 1000 ms/s, so the sum is 2000 ms/s.
        assert numpy.add(r, 1).magnitude == pytest.approx(2000.0)
        assert numpy.add(1, r).magnitude == pytest.approx(2000.0)
        assert numpy.maximum(ureg.Quantity(2, "ms") / ureg.Quantity(1, "s"), 1).to(
            "dimensionless"
        ).magnitude == pytest.approx(1.0)

    def test_plain_dimensionless_bare_number(self, ureg: UnitRegistry) -> None:
        assert numpy.add(
            ureg.Quantity(0.5, "dimensionless"), 0.5
        ).magnitude == pytest.approx(1.0)

    def test_scaled_dimensionless_comparison_with_bare(
        self,
        ureg: UnitRegistry,
    ) -> None:
        r = ureg.Quantity(2, "ms") / ureg.Quantity(1, "s")  # 0.002 dimensionless
        assert bool(numpy.less(r, 1))
        assert bool(numpy.equal(ureg.Quantity(1000, "ms") / ureg.Quantity(1, "s"), 1))

    @pytest.mark.parametrize("func", [numpy.mod, numpy.fmod, numpy.remainder])
    def test_mod_family_keeps_unit_with_bare_number(
        self,
        ureg: UnitRegistry,
        func: object,
    ) -> None:
        result = func(ureg.Quantity(5, "m"), 2)  # type: ignore[operator]
        assert result.magnitude == pytest.approx(1.0)
        assert result.dimensionality == ureg.Quantity(1, "m").dimensionality

    @pytest.mark.parametrize("func", [numpy.mod, numpy.fmod, numpy.remainder])
    def test_mod_family_uses_raw_magnitudes_no_conversion(
        self,
        ureg: UnitRegistry,
        func: object,
    ) -> None:
        # pint applies mod/fmod/remainder to raw magnitudes (no unit conversion)
        # and keeps the left operand's unit: 5 m mod 200 cm -> 5 m (not 1 m).
        result = func(ureg.Quantity(5.0, "m"), ureg.Quantity(200.0, "cm"))  # type: ignore[operator]
        assert result.magnitude == pytest.approx(5.0)
        assert result.dimensionality == ureg.Quantity(1, "m").dimensionality


class TestPowerAndMisc:
    """float_power, heaviside, and power array-exponent rules."""

    def test_float_power_behaves_like_power(self, ureg: UnitRegistry) -> None:
        result = numpy.float_power(ureg.Quantity(2, "m"), 3)
        assert result.magnitude == pytest.approx(8.0)
        assert result.dimensionality == ureg.Quantity(1, "m**3").dimensionality

    def test_float_power_scaled_dimensionless_exponent(
        self,
        ureg: UnitRegistry,
    ) -> None:
        exp = ureg.Quantity(1000, "ms") / ureg.Quantity(1, "s")
        result = numpy.float_power(ureg.Quantity(2, "m"), exp)
        assert result.magnitude == pytest.approx(2.0)

    def test_power_array_exponent_on_dimensional_base_raises(
        self,
        ureg: UnitRegistry,
    ) -> None:
        from pintrs import DimensionalityError

        with pytest.raises(DimensionalityError):
            numpy.power(ureg.Quantity(2, "m"), numpy.array([1, 2]))

    def test_power_array_exponent_on_dimensionless_base(
        self,
        ureg: UnitRegistry,
    ) -> None:
        result = numpy.power(ureg.Quantity(2, "dimensionless"), numpy.array([2, 3]))
        numpy.testing.assert_allclose(result.magnitude, [4.0, 8.0])
        assert result.dimensionless

    def test_power_dimensionless_base_with_array_quantity_exponent(
        self,
        ureg: UnitRegistry,
    ) -> None:
        # Regression: scalar dimensionless base with a Rust array exponent used
        # to crash with AttributeError.
        exp = ureg.Quantity(numpy.array([1000.0, 2000.0]), "ms") / ureg.Quantity(1, "s")
        result = numpy.power(ureg.Quantity(2, "dimensionless"), exp)
        numpy.testing.assert_allclose(result.magnitude, [2.0, 4.0])
        assert result.dimensionless

    def test_heaviside_is_dimensionless(self, ureg: UnitRegistry) -> None:
        assert numpy.heaviside(ureg.Quantity(2, "m"), 1).magnitude == pytest.approx(1.0)
        assert numpy.heaviside(
            ureg.Quantity(-3, "dimensionless"), 0.5
        ).magnitude == pytest.approx(0.0)

    def test_power_scaled_dimensionless_base(self, ureg: UnitRegistry) -> None:
        # (2 ms/s) == 0.002, so (2 ms/s) ** [1, 2] == [0.002, 4e-6].
        base = ureg.Quantity(2, "ms") / ureg.Quantity(1, "s")
        result = numpy.power(base, numpy.array([1.0, 2.0]))
        numpy.testing.assert_allclose(result.magnitude, [0.002, 4e-6])
        assert result.dimensionless


class TestUnitPreservingMisc:
    """Other ufuncs whose output units must match pint."""

    def test_bare_numerator_divide_is_inverse_unit(
        self,
        ureg: UnitRegistry,
    ) -> None:
        result = numpy.divide(3, ureg.Quantity(2, "m"))
        assert result.magnitude == pytest.approx(1.5)
        assert result.dimensionality == ureg.Quantity(1, "1/m").dimensionality

    def test_sign_strips_units(self, ureg: UnitRegistry) -> None:
        result = numpy.sign(ureg.Quantity(-3, "m"))
        assert result.magnitude == pytest.approx(-1.0)
        assert result.dimensionless

    def test_around_keeps_unit(self, ureg: UnitRegistry) -> None:
        result = numpy.around(ureg.Quantity(3.456, "m"), 1)
        assert result.magnitude == pytest.approx(3.5)
        assert result.dimensionality == ureg.Quantity(1, "m").dimensionality

    def test_round_keeps_unit(self, ureg: UnitRegistry) -> None:
        result = numpy.round(ureg.Quantity(3.456, "m"), 1)
        assert result.magnitude == pytest.approx(3.5)

    def test_modf_keeps_unit_on_both_parts(self, ureg: UnitRegistry) -> None:
        frac, integer = numpy.modf(ureg.Quantity(numpy.array([1.5, 2.25]), "m"))
        numpy.testing.assert_allclose(frac.magnitude, [0.5, 0.25])
        numpy.testing.assert_allclose(integer.magnitude, [1.0, 2.0])
        dim = ureg.Quantity(1, "m").dimensionality
        assert frac.dimensionality == dim
        assert integer.dimensionality == dim

    def test_frexp_mantissa_keeps_unit_exponent_dimensionless(
        self,
        ureg: UnitRegistry,
    ) -> None:
        mantissa, exponent = numpy.frexp(ureg.Quantity(1.5, "m"))
        assert mantissa.magnitude == pytest.approx(0.75)
        assert mantissa.dimensionality == ureg.Quantity(1, "m").dimensionality
        assert exponent.dimensionless

    def test_fmax_fmin_convert_units(self, ureg: UnitRegistry) -> None:
        a = ureg.Quantity(numpy.array([1.0, 2.0]), "m")
        b = ureg.Quantity(numpy.array([200.0, 50.0]), "cm")
        numpy.testing.assert_allclose(numpy.fmax(a, b).magnitude, [2.0, 2.0])
        numpy.testing.assert_allclose(numpy.fmin(a, b).magnitude, [1.0, 0.5])
