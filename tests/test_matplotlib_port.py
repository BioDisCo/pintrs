"""Matplotlib integration tests - ported from pint's test_matplotlib.py.

Tests setup_matplotlib and basic plotting with quantities.
"""

from __future__ import annotations

import pytest
from pintrs import UnitRegistry

plt = pytest.importorskip("matplotlib.pyplot", reason="matplotlib is not available")
np = pytest.importorskip("numpy", reason="numpy is not available")


@pytest.fixture
def ureg() -> UnitRegistry:
    ureg = UnitRegistry()
    ureg.setup_matplotlib(True)  # type: ignore[attr-defined]
    return ureg


@pytest.fixture(autouse=True)
def _use_agg_backend() -> None:
    """Force non-interactive backend."""
    plt.switch_backend("agg")


class TestSetupMatplotlib:
    def test_setup_does_not_raise(self) -> None:
        ureg = UnitRegistry()
        ureg.setup_matplotlib(True)  # type: ignore[attr-defined]

    def test_disable_does_not_raise(self) -> None:
        ureg = UnitRegistry()
        ureg.setup_matplotlib(True)  # type: ignore[attr-defined]
        ureg.setup_matplotlib(False)  # type: ignore[attr-defined]

    def test_setup_via_module(self) -> None:
        from pintrs.matplotlib_support import setup_matplotlib

        setup_matplotlib(True)
        setup_matplotlib(False)


class TestBasicPlotting:
    def test_plot_quantity_arrays(self, ureg: UnitRegistry) -> None:
        from pintrs.numpy_support import ArrayQuantity

        y = ArrayQuantity(np.linspace(0, 30), "meter", ureg)
        x = ArrayQuantity(np.linspace(0, 5), "second", ureg)

        fig, ax = plt.subplots()
        ax.plot(x.magnitude, y.magnitude, "tab:blue")
        plt.close(fig)

    def test_plot_creates_figure(self, ureg: UnitRegistry) -> None:
        fig, ax = plt.subplots()
        x_data = np.linspace(0, 10)
        y_data = np.linspace(0, 100)
        ax.plot(x_data, y_data)
        assert fig is not None
        plt.close(fig)

    def test_axhline_with_scalar(self, ureg: UnitRegistry) -> None:
        fig, ax = plt.subplots()
        ax.axhline(5.0, color="tab:red")
        assert fig is not None
        plt.close(fig)

    def test_axvline_with_scalar(self, ureg: UnitRegistry) -> None:
        fig, ax = plt.subplots()
        ax.axvline(2.0, color="tab:green")
        assert fig is not None
        plt.close(fig)


class TestPlotWithConversions:
    def test_plot_converted_values(self, ureg: UnitRegistry) -> None:
        distance = np.linspace(0, 30) * ureg.Quantity(1, "mile")
        time = np.linspace(0, 5) * ureg.Quantity(1, "hour")

        fig, ax = plt.subplots()
        d_km = [d.to("kilometer").magnitude for d in distance]
        t_min = [t.to("minute").magnitude for t in time]
        ax.plot(t_min, d_km, "tab:blue")
        plt.close(fig)

    def test_unit_conversion_for_axis(self, ureg: UnitRegistry) -> None:
        q_miles = ureg.Quantity(26400, "feet")
        q_km = q_miles.to("kilometer")
        assert q_km.magnitude > 0

        q_min = ureg.Quantity(120, "minutes")
        q_hr = q_min.to("hour")
        assert abs(q_hr.magnitude - 2.0) < 1e-10


class TestPlotCleanup:
    def test_close_all(self) -> None:
        fig, _ax = plt.subplots()
        plt.close(fig)
        plt.close("all")
