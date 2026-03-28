from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pintrs import UnitRegistry
from pintrs.context import Context

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def ureg() -> UnitRegistry:
    """Shared unit registry for tests."""
    return UnitRegistry()


@pytest.fixture(autouse=True)
def _clean_context_registry() -> Iterator[None]:
    """Reset Context._REGISTRY between tests to prevent leakage."""
    saved = dict(Context._REGISTRY)
    yield
    Context._REGISTRY.clear()
    Context._REGISTRY.update(saved)
