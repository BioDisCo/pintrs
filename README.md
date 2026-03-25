# pintrs

Fast physical units for Python, with a `pint`-compatible API and Rust performance.

`pintrs` is for people who like [`pint`](https://pint.readthedocs.io/)'s ergonomics but not its runtime cost. It keeps the familiar `UnitRegistry` and `Quantity` workflow, with the hot path implemented in Rust.

- Drop into common `pint` workflows with minimal code changes
- Usually **7-100x faster** on core operations
- Works with NumPy, pandas, Babel, measurements, contexts, groups, and systems

## Why pintrs

If your code spends real time creating quantities, parsing unit strings, or converting units, `pintrs` removes a lot of overhead without asking you to relearn the API.

- Quantity creation: **10x faster**
- Parsing expressions like `"9.81 m/s**2"`: **104x faster**
- Common conversions like `km/h -> m/s`: **8x faster**

Benchmarks below were measured on this branch with Python 3.13.5. Lower is better.

| Operation | pintrs | pint | Speedup |
|---|--:|--:|--:|
| Quantity creation | 0.35 us | 3.65 us | **10x** |
| Parse string (`"9.81 m/s**2"`) | 0.67 us | 70.12 us | **104x** |
| Conversion (km -> m) | 1.18 us | 7.84 us | **7x** |
| Conversion (km/h -> m/s) | 1.75 us | 13.80 us | **8x** |
| Addition (compatible units) | 0.88 us | 12.01 us | **14x** |
| Multiply by scalar | 0.13 us | 5.88 us | **46x** |
| Multiply quantities | 0.16 us | 5.44 us | **33x** |
| Parse units (`"kg * m / s ** 2"`) | 0.88 us | 23.66 us | **27x** |
| String formatting | 0.29 us | 8.41 us | **29x** |

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
