# pintrs

A fast, Rust-powered drop-in replacement for [pint](https://pint.readthedocs.io/) -- the Python physical units library.

pintrs reimplements pint's core unit registry and quantity system in Rust via [PyO3](https://pyo3.rs/), providing the same Python API with significantly better performance for unit parsing, conversion, and arithmetic.

## Features

- **Drop-in replacement**: Same API as pint for `UnitRegistry`, `Quantity`, `Unit`, and common operations
- **Fast**: Rust-native unit parsing, registry lookup, and conversion factor computation
- **NumPy support**: `ArrayQuantity` wraps numpy arrays with unit tracking and full ufunc integration
- **Type-safe**: Full type stubs (`.pyi`) for mypy and pyright in strict mode
- **Measurement support**: `Measurement` class for quantities with uncertainty propagation
- **Compatibility stubs**: `Context`, `Group`, `System` classes for code that references pint's advanced features

## Installation

### From source (requires Rust toolchain and maturin)

```bash
pip install maturin
maturin develop --release
```

### Running tests

```bash
pip install pytest numpy
pytest
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

# Error propagation
m2 = Measurement(Quantity(50.0, "meter"), 0.3)
print(m + m2)  # adds in quadrature
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

## Architecture

```
pintrs/
  src/              # Rust source (PyO3 extension module)
    lib.rs          # Module entry point
    registry.rs     # UnitRegistry: parsing, storage, conversion
    definition.rs   # Unit definition parsing
    parser.rs       # Expression parser
    errors.rs       # Error types
    units_container.rs
  python/pintrs/    # Python package
    __init__.py     # Public API, Measurement, Context/Group/System stubs
    _core.pyi       # Type stubs for the Rust extension
    numpy_support.py # ArrayQuantity with numpy ufunc support
  tests/            # pytest suite
```

## Compatibility with pint

pintrs targets API compatibility with pint's most-used features. Supported:

- `UnitRegistry`, `Quantity`, `Unit` with full arithmetic
- Unit parsing, conversion, base/root/compact/reduced/preferred units
- `__getattr__` on registry (`ureg.meter`, `ureg.speed_of_light`)
- Serialization via `__reduce__` (pickle) and `to_tuple`/`from_tuple`
- `wraps` and `check` decorators
- `Measurement` with uncertainty propagation
- `format_babel` stubs
- Context/Group/System stubs (API-compatible, no-op)

Not yet implemented:

- Full context-based conversions (spectroscopy, etc.)
- Babel/locale formatting
- Logarithmic units
- pandas ExtensionArray integration

## Development

```bash
# Build
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
