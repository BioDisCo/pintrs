"""Benchmarks comparing pintrs vs pint performance."""

from __future__ import annotations

import sys
import timeit

ITERATIONS = 10_000
REPEAT = 3


def bench(label: str, stmt: str, setup: str, number: int = ITERATIONS) -> float:
    """Run a benchmark and print results. Returns best time per iteration in us."""
    times = timeit.repeat(stmt, setup=setup, number=number, repeat=REPEAT)
    best = min(times) / number * 1e6
    print(f"  {label:.<50s} {best:>8.2f} us")
    return best


def run_pintrs_benchmarks() -> dict[str, float]:
    """Run all pintrs benchmarks."""
    print("pintrs benchmarks")
    print("=" * 62)
    results: dict[str, float] = {}

    setup = "from pintrs import UnitRegistry; ureg = UnitRegistry()"

    results["create_quantity"] = bench(
        "Quantity creation",
        "ureg.Quantity(1.0, 'meter')",
        setup,
    )

    results["create_quantity_str"] = bench(
        "Quantity from string",
        "ureg.Quantity('9.81 m/s**2')",
        setup,
    )

    results["conversion"] = bench(
        "Unit conversion (km -> m)",
        "q.to('meter')",
        setup + "; q = ureg.Quantity(1.0, 'kilometer')",
    )

    results["conversion_complex"] = bench(
        "Unit conversion (km/h -> m/s)",
        "q.to('m/s')",
        setup + "; q = ureg.Quantity(100.0, 'km/hr')",
    )

    results["add_same"] = bench(
        "Addition (same units)",
        "a + b",
        setup + "; a = ureg.Quantity(1.0, 'meter'); b = ureg.Quantity(2.0, 'meter')",
    )

    results["add_compatible"] = bench(
        "Addition (compatible units)",
        "a + b",
        setup + "; a = ureg.Quantity(1.0, 'km'); b = ureg.Quantity(500.0, 'meter')",
    )

    results["mul_scalar"] = bench(
        "Multiply by scalar",
        "q * 3.0",
        setup + "; q = ureg.Quantity(1.0, 'meter')",
    )

    results["mul_quantity"] = bench(
        "Multiply quantities",
        "a * b",
        setup + "; a = ureg.Quantity(2.0, 'meter'); b = ureg.Quantity(3.0, 'second')",
    )

    results["div_quantity"] = bench(
        "Divide quantities",
        "a / b",
        setup + "; a = ureg.Quantity(10.0, 'meter'); b = ureg.Quantity(2.0, 'second')",
    )

    results["magnitude"] = bench(
        "Property access (.magnitude)",
        "q.magnitude",
        setup + "; q = ureg.Quantity(1.0, 'meter')",
    )

    results["to_base"] = bench(
        "To base units",
        "q.to_base_units()",
        setup + "; q = ureg.Quantity(1.0, 'kilometer')",
    )

    results["parse_units"] = bench(
        "Parse units string",
        "ureg.parse_units('kg * m / s ** 2')",
        setup,
    )

    results["getattr"] = bench(
        "Registry getattr (ureg.meter)",
        "ureg.meter",
        setup,
    )

    results["comparison"] = bench(
        "Comparison (>)",
        "a > b",
        setup + "; a = ureg.Quantity(2.0, 'meter'); b = ureg.Quantity(1.0, 'meter')",
    )

    results["format"] = bench(
        "String formatting",
        "str(q)",
        setup + "; q = ureg.Quantity(9.81, 'meter / second ** 2')",
    )

    return results


def run_pint_benchmarks() -> dict[str, float]:
    """Run all pint benchmarks."""
    print("\npint benchmarks")
    print("=" * 62)
    results: dict[str, float] = {}

    setup = "import pint; ureg = pint.UnitRegistry(cache_folder=None)"

    results["create_quantity"] = bench(
        "Quantity creation",
        "ureg.Quantity(1.0, 'meter')",
        setup,
    )

    results["create_quantity_str"] = bench(
        "Quantity from string",
        "ureg.Quantity('9.81 m/s**2')",
        setup,
    )

    results["conversion"] = bench(
        "Unit conversion (km -> m)",
        "q.to('meter')",
        setup + "; q = ureg.Quantity(1.0, 'kilometer')",
    )

    results["conversion_complex"] = bench(
        "Unit conversion (km/h -> m/s)",
        "q.to('m/s')",
        setup + "; q = ureg.Quantity(100.0, 'km/hr')",
    )

    results["add_same"] = bench(
        "Addition (same units)",
        "a + b",
        setup + "; a = ureg.Quantity(1.0, 'meter'); b = ureg.Quantity(2.0, 'meter')",
    )

    results["add_compatible"] = bench(
        "Addition (compatible units)",
        "a + b",
        setup + "; a = ureg.Quantity(1.0, 'km'); b = ureg.Quantity(500.0, 'meter')",
    )

    results["mul_scalar"] = bench(
        "Multiply by scalar",
        "q * 3.0",
        setup + "; q = ureg.Quantity(1.0, 'meter')",
    )

    results["mul_quantity"] = bench(
        "Multiply quantities",
        "a * b",
        setup + "; a = ureg.Quantity(2.0, 'meter'); b = ureg.Quantity(3.0, 'second')",
    )

    results["div_quantity"] = bench(
        "Divide quantities",
        "a / b",
        setup + "; a = ureg.Quantity(10.0, 'meter'); b = ureg.Quantity(2.0, 'second')",
    )

    results["magnitude"] = bench(
        "Property access (.magnitude)",
        "q.magnitude",
        setup + "; q = ureg.Quantity(1.0, 'meter')",
    )

    results["to_base"] = bench(
        "To base units",
        "q.to_base_units()",
        setup + "; q = ureg.Quantity(1.0, 'kilometer')",
    )

    results["parse_units"] = bench(
        "Parse units string",
        "ureg.parse_units('kg * m / s ** 2')",
        setup,
    )

    results["getattr"] = bench(
        "Registry getattr (ureg.meter)",
        "ureg.meter",
        setup,
    )

    results["comparison"] = bench(
        "Comparison (>)",
        "a > b",
        setup + "; a = ureg.Quantity(2.0, 'meter'); b = ureg.Quantity(1.0, 'meter')",
    )

    results["format"] = bench(
        "String formatting",
        "str(q)",
        setup + "; q = ureg.Quantity(9.81, 'meter / second ** 2')",
    )

    return results


def print_comparison(
    pintrs_results: dict[str, float],
    pint_results: dict[str, float],
) -> None:
    """Print a side-by-side comparison."""
    print("\nComparison (speedup = pint / pintrs)")
    print("=" * 72)
    print(f"  {'Benchmark':<40s} {'pintrs':>8s} {'pint':>8s} {'speedup':>8s}")
    print("-" * 72)
    for key, p in pintrs_results.items():
        t = pint_results.get(key, 0)
        speedup = t / p if p > 0 else 0
        notable_threshold = 5
        marker = " <<" if speedup > notable_threshold else ""
        print(f"  {key:<40s} {p:>7.2f}u {t:>7.2f}u {speedup:>7.1f}x{marker}")


def numpy_benchmarks() -> None:
    """Run numpy array quantity benchmarks."""
    try:
        import importlib.util

        if importlib.util.find_spec("numpy") is None:
            raise ImportError
    except ImportError:
        print("\nNumPy not installed, skipping array benchmarks")
        return

    print("\nNumPy array benchmarks (pintrs.ArrayQuantity)")
    print("=" * 62)

    setup = (
        "import numpy as np; "
        "from pintrs import UnitRegistry; "
        "from pintrs.numpy_support import ArrayQuantity; "
        "ureg = UnitRegistry(); "
        "arr = np.random.rand(1000); "
        "q = ArrayQuantity(arr, 'meter', ureg)"
    )

    bench(
        "Create ArrayQuantity (1000 elements)",
        "ArrayQuantity(arr, 'meter', ureg)",
        setup,
    )
    bench("Convert units (1000 elements)", "q.to('kilometer')", setup)
    bench("Add arrays (1000 elements)", "q + q", setup)
    bench("Multiply by scalar (1000 elements)", "q * 2.0", setup)
    bench("np.sqrt (1000 elements)", "np.sqrt(q * q)", setup)


def main() -> None:
    """Run all benchmarks."""
    print(f"Python {sys.version}")
    print(f"Iterations: {ITERATIONS}, Repeats: {REPEAT}\n")

    pintrs_results = run_pintrs_benchmarks()

    has_pint = False
    pint_results: dict[str, float] = {}
    try:
        import pint  # noqa: F401

        has_pint = True
    except ImportError:
        pass

    if has_pint:
        pint_results = run_pint_benchmarks()
        print_comparison(pintrs_results, pint_results)
    else:
        print("\npint not installed -- skipping comparison benchmarks")

    numpy_benchmarks()


if __name__ == "__main__":
    main()
