# pintrs

Fast physical units for Python, with a `pint`-compatible API and Rust performance.

`pintrs` is for people who like [`pint`](https://pint.readthedocs.io/)'s ergonomics but not its runtime cost. It keeps the familiar `UnitRegistry` and `Quantity` workflow, with the hot path implemented in Rust.

- Drop into common `pint` workflows with minimal code changes
- Usually **8-150x faster** on core operations
- Works with NumPy, pandas, Babel, measurements, contexts, groups, and systems

## Why pintrs

If your code spends real time creating quantities, parsing unit strings, or converting units, `pintrs` removes a lot of overhead without asking you to relearn the API.

- Quantity creation: **9x faster**
- Parsing unit strings: **152x faster**
- Same-unit addition: **14x faster**

Benchmarks below were measured with Python 3.12. Lower is better.

| Operation | pintrs | pint | Speedup |
|---|--:|--:|--:|
| Quantity creation | 0.37 us | 3.33 us | **9x** |
| Parse string (`"9.81 m/s**2"`) | 0.55 us | 50.21 us | **91x** |
| Conversion (km -> m) | 0.94 us | 7.43 us | **8x** |
| Conversion (km/h -> m/s) | 1.71 us | 13.30 us | **8x** |
| Addition (same units) | 0.34 us | 4.64 us | **14x** |
| Addition (compatible units) | 1.03 us | 11.56 us | **11x** |
| Multiply by scalar | 0.25 us | 5.31 us | **21x** |
| Multiply quantities | 0.38 us | 5.01 us | **13x** |
| Comparison (>) | 0.12 us | 1.16 us | **10x** |
| To base units | 0.37 us | 6.43 us | **17x** |
| Parse units (`"kg * m / s ** 2"`) | 0.26 us | 38.85 us | **152x** |
| String formatting | 0.44 us | 7.20 us | **16x** |

Run `python examples/benchmark.py` to reproduce the numbers. Install `pint` alongside `pintrs` for the comparison run.

## Migrating from pint

If you already use `pint`, the change is intentionally small: replace `pint` with `pintrs` in your dependencies and swap your imports.

```diff
- pint
+ pintrs
```

```diff
- from pint import UnitRegistry
+ from pintrs import UnitRegistry

ureg = UnitRegistry()
```

Your existing quantity code should continue to look like `pint` code:

```python
distance = 5 * ureg.kilometer
time = 2 * ureg.hour
speed = distance / time

print(speed)           # 2.5 kilometer / hour
print(speed.to("m/s")) # 0.6944... meter / second
```

## Compatibility with pint

`pintrs` targets full API compatibility with `pint`.

That includes the core registry and quantity model, conversions and formatting, decorators, measurements, contexts, groups, systems, and integrations with NumPy, pandas, and Babel.

If you already have working `pint` code and performance is the problem, `pintrs` is designed to be the least disruptive upgrade path.

## Installation

```bash
pip install pintrs
```

NumPy, pandas, and Babel integrations are available when those packages are installed.

## What you get

- The familiar `pint` API, with Rust underneath
- Substantial speedups on quantity creation, parsing, conversion, arithmetic, and formatting
- Support for NumPy, pandas, Babel, measurements, contexts, groups, systems, logarithmic units, and decorators
- Type information for mypy and pyright
