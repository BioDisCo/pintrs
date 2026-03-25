# pintrs

A fast, Rust-powered drop-in replacement for [pint](https://pint.readthedocs.io/) -- the Python physical units library.

pintrs reimplements pint's core unit registry and quantity system in Rust via [PyO3](https://pyo3.rs/), giving you the same Python API with significantly better performance.

## Installation

```bash
pip install pintrs
```

Optional integrations:

```bash
pip install "pintrs[numpy]"
pip install "pintrs[pandas]"
pip install "pintrs[babel]"
pip install "pintrs[all]"
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

pintrs is typically **7-100x faster** than pint on common operations. Benchmarks below were measured on this branch with Python 3.13.5 (lower is better):

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

Run `python examples/benchmark.py` to reproduce (install `pint` for comparison).

## Features

- **Drop-in replacement** for pint's `UnitRegistry`, `Quantity`, `Unit`, and common operations
- **NumPy support** via `ArrayQuantity` with full ufunc integration
- **Type-safe** with full `.pyi` stubs for mypy and pyright in strict mode
- **Measurement support** for quantities with uncertainty propagation
- **Context**, **Group**, and **System** support for context-based conversions, unit collections, and coherent unit systems

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
- Context-based conversions (spectroscopy, Boltzmann, chemistry)
- Group (named unit collections: imperial, metric, cgs, US_customary)
- System (coherent unit sets: mks/SI, cgs, imperial, Gaussian, atomic)
- Babel/locale formatting (`format_babel`)
- Logarithmic units (`LogarithmicQuantity` for dB/dBm/dBW/Np/Bel)
- Pandas `ExtensionArray` integration (`PintArray`/`PintDtype`)

## Development

```bash
# Build (requires Rust toolchain and maturin)
maturin develop --release

# Lint and format
ruff check --fix python/ tests/
ruff format python/ tests/

# Type check
mypy python/pintrs/
pyright python/pintrs/

# Test
pytest
```

## License

Apache-2.0
