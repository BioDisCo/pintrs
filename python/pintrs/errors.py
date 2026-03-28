"""pintrs exception classes.

Re-exports from the Rust core for convenience:
    from pintrs.errors import DimensionalityError
"""

from __future__ import annotations

from pintrs._core import (
    DefinitionSyntaxError,
    DimensionalityError,
    OffsetUnitCalculusError,
    PintError,
    RedefinitionError,
    UndefinedUnitError,
)

__all__ = [
    "DefinitionSyntaxError",
    "DimensionalityError",
    "OffsetUnitCalculusError",
    "PintError",
    "RedefinitionError",
    "UndefinedUnitError",
]
