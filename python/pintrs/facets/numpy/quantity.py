"""Pint-compatible NumpyQuantity alias.

Re-exports pintrs.Quantity as NumpyQuantity for compatibility with code
that imports from pint.facets.numpy.quantity.
"""

from __future__ import annotations

from pintrs import Quantity as NumpyQuantity

__all__ = ["NumpyQuantity"]
