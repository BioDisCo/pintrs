from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_import_pintrs_without_numpy_or_pandas() -> None:
    script = """
import builtins
import sys

orig_import = builtins.__import__

def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "numpy" or name.startswith("numpy."):
        raise ImportError("blocked numpy")
    if name == "pandas" or name.startswith("pandas."):
        raise ImportError("blocked pandas")
    return orig_import(name, globals, locals, fromlist, level)

builtins.__import__ = blocked_import

import pintrs
assert pintrs.UnitRegistry is not None

try:
    pintrs.ArrayQuantity([1, 2, 3], "meter")
except ModuleNotFoundError as exc:
    assert "numpy" in str(exc).lower()
else:
    raise AssertionError("ArrayQuantity should require numpy")

try:
    pintrs.PintDtype("meter")
except ModuleNotFoundError as exc:
    assert "pandas" in str(exc).lower()
else:
    raise AssertionError("PintDtype should require pandas")
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", script],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "python")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
