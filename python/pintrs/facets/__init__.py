"""Pint-compatible facets module.

Provides the same import paths as pint.facets for drop-in compatibility.
"""

from __future__ import annotations

from pintrs.facets import numpy, plain

__all__ = ["numpy", "plain"]
