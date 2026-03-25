"""Type stubs for the pintrs package."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pintrs._core import (
    DefinitionSyntaxError as DefinitionSyntaxError,
)
from pintrs._core import (
    DimensionalityError as DimensionalityError,
)
from pintrs._core import (
    OffsetUnitCalculusError as OffsetUnitCalculusError,
)
from pintrs._core import (
    PintError as PintError,
)
from pintrs._core import (
    Quantity as Quantity,
)
from pintrs._core import (
    RedefinitionError as RedefinitionError,
)
from pintrs._core import (
    UndefinedUnitError as UndefinedUnitError,
)
from pintrs._core import (
    Unit as Unit,
)
from pintrs._core import (
    UnitRegistry as UnitRegistry,
)
from pintrs.context import Context as Context
from pintrs.group import Group as Group
from pintrs.logarithmic import LogarithmicQuantity as LogarithmicQuantity
from pintrs.numpy_support import ArrayQuantity as ArrayQuantity
from pintrs.system import System as System

__version__: str
__all__: list[str]

def set_application_registry(ureg: UnitRegistry) -> None: ...
def get_application_registry() -> UnitRegistry: ...
def wraps(
    ureg: UnitRegistry,
    ret: str | Unit | None,
    args: tuple[str | Unit | None, ...] | None = None,
    strict: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...
def check(
    ureg: UnitRegistry,
    *args: str | None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]: ...
def make_quantity(
    ureg: UnitRegistry,
    value: Any,
    units: str | None = None,
) -> Any: ...

class Measurement:
    def __init__(
        self,
        value: Quantity | float,
        error: Quantity | float,
        units: str | None = None,
    ) -> None: ...
    @property
    def value(self) -> Quantity: ...
    @property
    def error(self) -> Quantity: ...
    @property
    def magnitude(self) -> float: ...
    @property
    def units(self) -> Unit: ...
    @property
    def rel(self) -> float: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __add__(self, other: Measurement) -> Measurement: ...
    def __sub__(self, other: Measurement) -> Measurement: ...
    def __mul__(self, other: Measurement | float) -> Measurement: ...
    def __truediv__(self, other: Measurement | float) -> Measurement: ...
