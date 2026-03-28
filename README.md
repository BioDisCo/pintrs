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

**Scalar operations:**

| Operation | pintrs | pint | Speedup |
|---|--:|--:|--:|
| Parse expression | 0.6 us | 140 us | **219x** |
| Conversion (`q.to("km")`) | 1.0 us | 46 us | **46x** |
| Multiply by scalar | 0.3 us | 7.0 us | **25x** |
| Quantity creation | 0.5 us | 7.7 us | **16x** |
| To base units | 0.4 us | 3.4 us | **8x** |
| Addition (same units) | 0.5 us | 4.5 us | **8x** |
| Multiply quantities | 0.8 us | 5.5 us | **7x** |

**NumPy arrays (1000 elements):**

| Operation | pintrs | pint | Speedup |
|---|--:|--:|--:|
| Conversion (`arr.to("km")`) | 1.7 us | 48 us | **28x** |
| Sum | 1.2 us | 23 us | **19x** |
| Create | 1.2 us | 8.2 us | **7x** |
| Multiply by scalar | 1.0 us | 6.9 us | **7x** |
| Addition | 0.8 us | 5.4 us | **7x** |

## Something not working?

If you hit a compatibility issue or something behaves differently from `pint`, please [open an issue](https://github.com/BioDisCo/pintrs/issues). Migration should be painless, and if it isn't, we want to know about it.
