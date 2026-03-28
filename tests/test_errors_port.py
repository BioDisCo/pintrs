"""Ported from pint's test_errors.py - comprehensive exception tests."""

from __future__ import annotations

import pickle

import pytest
from pintrs import (
    DefinitionSyntaxError,
    DimensionalityError,
    OffsetUnitCalculusError,
    PintError,
    RedefinitionError,
    UndefinedUnitError,
    UnitRegistry,
)


class TestExceptionHierarchy:
    """All pint exceptions inherit from PintError which inherits from ValueError."""

    def test_pint_error_is_value_error(self) -> None:
        assert issubclass(PintError, ValueError)

    def test_dimensionality_error_is_pint_error(self) -> None:
        assert issubclass(DimensionalityError, PintError)

    def test_undefined_unit_error_is_pint_error(self) -> None:
        assert issubclass(UndefinedUnitError, PintError)

    def test_offset_unit_calculus_error_is_pint_error(self) -> None:
        assert issubclass(OffsetUnitCalculusError, PintError)

    def test_definition_syntax_error_is_pint_error(self) -> None:
        assert issubclass(DefinitionSyntaxError, PintError)

    def test_redefinition_error_is_pint_error(self) -> None:
        assert issubclass(RedefinitionError, PintError)

    def test_dimensionality_error_is_value_error(self) -> None:
        assert issubclass(DimensionalityError, ValueError)

    def test_undefined_unit_error_is_value_error(self) -> None:
        assert issubclass(UndefinedUnitError, ValueError)


class TestExceptionRaising:
    """Exceptions are raised in the correct situations."""

    def test_incompatible_conversion_raises_dimensionality_error(self) -> None:
        ureg = UnitRegistry()
        with pytest.raises(DimensionalityError):
            ureg.Quantity(1, "meter").to("second")

    def test_unknown_unit_raises_undefined_unit_error(self) -> None:
        ureg = UnitRegistry()
        with pytest.raises(UndefinedUnitError):
            ureg.Quantity(1, "gibberish_unit")

    def test_redefinition_is_silently_ignored(self) -> None:
        ureg = UnitRegistry()
        ureg.define("meter = [length]")  # should not raise (pint compat)

    def test_bad_definition_raises(self) -> None:
        ureg = UnitRegistry()
        with pytest.raises((DefinitionSyntaxError, ValueError)):
            ureg.define("= = = = =")

    def test_catch_as_pint_error(self) -> None:
        ureg = UnitRegistry()
        with pytest.raises(PintError):
            ureg.Quantity(1, "meter").to("second")

    def test_catch_as_value_error(self) -> None:
        ureg = UnitRegistry()
        with pytest.raises(ValueError):
            ureg.Quantity(1, "meter").to("second")


class TestExceptionPickle:
    """Exceptions can be pickled and unpickled."""

    def test_pickle_dimensionality_error(self) -> None:
        ureg = UnitRegistry()
        try:
            ureg.Quantity(1, "meter").to("second")
        except DimensionalityError as e:
            restored = pickle.loads(pickle.dumps(e))
            assert isinstance(restored, DimensionalityError)

    def test_pickle_undefined_unit_error(self) -> None:
        ureg = UnitRegistry()
        try:
            ureg.Quantity(1, "gibberish_unit_xyz")
        except UndefinedUnitError as e:
            restored = pickle.loads(pickle.dumps(e))
            assert isinstance(restored, UndefinedUnitError)


class TestExceptionMessages:
    """Exception messages are informative."""

    def test_dimensionality_error_mentions_units(self) -> None:
        ureg = UnitRegistry()
        with pytest.raises(DimensionalityError, match=r"meter|second|length|time"):
            ureg.Quantity(1, "meter").to("second")

    def test_undefined_unit_error_mentions_unit_name(self) -> None:
        ureg = UnitRegistry()
        with pytest.raises(UndefinedUnitError, match="nonexistent_xyz"):
            ureg.Quantity(1, "nonexistent_xyz")
