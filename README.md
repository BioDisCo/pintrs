# pintrs

If you like [`pint`](https://pint.readthedocs.io/) but wish it were faster, `pintrs` is for you. Same API, same workflow, but with the heavy lifting done in Rust.

## Installation

```bash
pip install pintrs
```

## Migration

`pintrs` is designed as a drop-in replacement. Swap your dependency, update your imports, and you're done.

```diff
- pint
+ pintrs
```

```diff
- from pint import UnitRegistry
+ from pintrs import UnitRegistry

ureg = UnitRegistry()
distance = 5 * ureg.kilometer
time = 2 * ureg.hour
speed = (distance / time).to("m/s")
```

## How much faster?

Here's what you get for that one-line change. Measured with Python 3.12; run `python examples/benchmark.py` to reproduce.

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

## Something not working?

If you hit a compatibility issue or something behaves differently from `pint`, please [open an issue](https://github.com/BioDisCo/pintrs/issues). Migration should be painless, and if it isn't, we want to know about it.
