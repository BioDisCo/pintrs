"""Testing utilities for pintrs.

Provides assertion helpers for comparing unit-aware quantities,
similar to pint.testing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pintrs._core import Quantity


def assert_equal(first: Quantity, second: Quantity) -> None:
    """Assert that two Quantities are exactly equal (magnitude and units).

    Args:
        first: First quantity.
        second: Second quantity.

    Raises:
        AssertionError: If quantities are not equal.
    """
    try:
        if str(first.units) != str(second.units):
            second = second.to(str(first.units))
    except Exception as exc:
        msg = f"{first} != {second}"
        raise AssertionError(msg) from exc
    if first.magnitude != second.magnitude:
        msg = f"{first} != {second}"
        raise AssertionError(msg)


def assert_allclose(
    first: Quantity,
    second: Quantity,
    rtol: float = 1e-7,
    atol: float = 0.0,
) -> None:
    """Assert that two Quantities are approximately equal.

    Converts to the same units before comparing magnitudes.

    Args:
        first: First quantity.
        second: Second quantity.
        rtol: Relative tolerance.
        atol: Absolute tolerance.

    Raises:
        AssertionError: If quantities are not close enough.
    """
    try:
        if str(first.units) != str(second.units):
            second = second.to(str(first.units))
    except Exception as exc:
        msg = f"{first} != {second} within rtol={rtol}, atol={atol}"
        raise AssertionError(msg) from exc
    a = first.magnitude
    b = second.magnitude
    if abs(a - b) > atol + rtol * abs(b):
        msg = (
            f"{first} != {second} within rtol={rtol}, atol={atol} "
            f"(diff={abs(a - b):.6e})"
        )
        raise AssertionError(msg)
