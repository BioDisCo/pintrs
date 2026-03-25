from __future__ import annotations

import pytest
from pintrs import UnitRegistry


@pytest.fixture
def ureg() -> UnitRegistry:
    """Shared unit registry for tests."""
    return UnitRegistry()
