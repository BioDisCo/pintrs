"""Pint-compatible plain facet.

Re-exports pintrs.Quantity as PlainQuantity for compatibility with code
that imports from pint.facets.plain.
"""

from __future__ import annotations

from pintrs import Quantity as PlainQuantity
from pintrs import Unit as PlainUnit

__all__ = ["PlainQuantity", "PlainUnit"]
