# pintrs

A fast, Rust-powered drop-in replacement for [pint](https://pint.readthedocs.io/) -- the Python physical units library.

pintrs reimplements pint's core unit registry and quantity system in Rust via [PyO3](https://pyo3.rs/), giving you the same Python API with significantly better performance.

## Installation

```bash
pip install pintrs
```

## Quick start

```python
from pintrs import UnitRegistry

ureg = UnitRegistry()

# Create quantities
distance = ureg.Quantity(5.0, "kilometer")
time = ureg.Quantity(2.0, "hour")

# Arithmetic with automatic unit tracking
speed = distance / time
print(speed)           # 2.5 kilometer / hour
print(speed.to("m/s")) # 0.6944... meter / second

# Attribute-style access
print(ureg.meter)      # 1 meter
print(ureg.speed_of_light)
```

## Performance

pintrs is **10-90x faster** than pint on common operations. Benchmarks on Python 3.13 (lower is better):

| Operation | pintrs | pint | Speedup |
|---|--:|--:|--:|
| Quantity creation | 0.35 us | 3.64 us | **10x** |
| Parse string (`"9.81 m/s**2"`) | 0.74 us | 70.61 us | **96x** |
| Conversion (km -> m) | 1.19 us | 8.08 us | **7x** |
| Conversion (km/h -> m/s) | 1.68 us | 14.11 us | **8x** |
| Addition (compatible units) | 0.94 us | 12.66 us | **13x** |
| Multiply by scalar | 0.13 us | 5.59 us | **41x** |
| Multiply quantities | 0.17 us | 5.38 us | **31x** |
| Parse units (`"kg * m / s ** 2"`) | 0.95 us | 23.66 us | **25x** |
| String formatting | 0.29 us | 8.58 us | **29x** |

Run `python examples/benchmark.py` to reproduce (install `pint` for comparison).

## Features

- **Drop-in replacement** for pint's `UnitRegistry`, `Quantity`, `Unit`, and common operations
- **NumPy support** via `ArrayQuantity` with full ufunc integration
- **Type-safe** with full `.pyi` stubs for mypy and pyright in strict mode
- **Measurement support** for quantities with uncertainty propagation
- **Compatibility stubs** for `Context`, `Group`, `System` so existing code doesn't break

## NumPy integration

```python
import numpy as np
from pintrs import UnitRegistry
from pintrs.numpy_support import ArrayQuantity

ureg = UnitRegistry()

distances = ArrayQuantity(np.array([1.0, 2.0, 3.0]), "kilometer", ureg)
result = distances.to("meter")
print(result.magnitude)  # [1000. 2000. 3000.]

# NumPy ufuncs work transparently
print(np.sqrt(ArrayQuantity(np.array([4.0, 9.0]), "m**2", ureg)))
```

## Measurements (uncertainty)

```python
from pintrs import Measurement, Quantity

m = Measurement(Quantity(100.0, "meter"), 0.5)
print(m)       # 100.0 +/- 0.5 meter
print(m.rel)   # 0.005

# Error propagation (adds in quadrature)
m2 = Measurement(Quantity(50.0, "meter"), 0.3)
print(m + m2)
```

## Decorators

```python
from pintrs import UnitRegistry, wraps, check

ureg = UnitRegistry()

@wraps(ureg, ret="meter/second", args=("meter", "second"))
def speed(distance, time):
    return distance / time

result = speed(ureg.Quantity(100, "km"), ureg.Quantity(2, "hour"))
print(result)  # in m/s

@check(ureg, "[length]", "[time]")
def velocity(d, t):
    return d / t
```

## Custom units

```python
ureg = UnitRegistry()
ureg.define("smoot = 1.7018 * meter")
print(ureg.Quantity(1, "smoot").to("meter"))  # 1.7018 meter
```

## Compatibility with pint

pintrs targets API compatibility with pint's most-used features:

- `UnitRegistry`, `Quantity`, `Unit` with full arithmetic
- Unit parsing, conversion, base/root/compact/reduced/preferred units
- `__getattr__` on registry (`ureg.meter`, `ureg.speed_of_light`)
- Serialization via `__reduce__` (pickle) and `to_tuple`/`from_tuple`
- `wraps` and `check` decorators
- `Measurement` with uncertainty propagation
- Context/Group/System stubs (API-compatible, no-op)

**Not yet implemented:** full context-based conversions (spectroscopy, etc.), Babel/locale formatting, logarithmic units, pandas ExtensionArray integration.

## Development

```bash
# Build (requires Rust toolchain and maturin)
maturin develop --release

# Lint and format
ruff check --fix . && ruff format .

# Type check
mypy python/pintrs/
pyright python/pintrs/

# Test
pytest
```

## License

Apache-2.0
