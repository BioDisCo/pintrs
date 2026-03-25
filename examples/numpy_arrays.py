"""NumPy array quantity support in pintrs."""

from __future__ import annotations

import numpy as np
from pintrs import UnitRegistry
from pintrs.numpy_support import ArrayQuantity

ureg = UnitRegistry()

# --- Creating array quantities ---

distances = ArrayQuantity(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), "kilometer", ureg)
print(f"Distances: {distances}")
print(f"Shape: {distances.shape}, ndim: {distances.ndim}, dtype: {distances.dtype}")

# --- Conversions ---

in_meters = distances.to("meter")
print(f"\nIn meters: {in_meters.magnitude}")

in_miles = distances.to("mile")
print(f"In miles: {in_miles.magnitude}")

# --- Arithmetic ---

times = ArrayQuantity(np.array([0.5, 1.0, 1.5, 2.0, 2.5]), "hour", ureg)
speeds = distances / times
print(f"\nSpeeds: {speeds}")

doubled = distances * 2
print(f"Doubled: {doubled.magnitude} {doubled.units}")

# --- NumPy ufuncs ---

areas = ArrayQuantity(np.array([4.0, 9.0, 16.0, 25.0]), "m**2", ureg)
sides = np.sqrt(areas)
print(f"\nSquare roots: {sides.magnitude} {sides.units}")

a = ArrayQuantity(np.array([1.0, 2.0, 3.0]), "meter", ureg)
b = ArrayQuantity(np.array([4.0, 5.0, 6.0]), "meter", ureg)
print(f"np.add: {np.add(a, b).magnitude}")
print(f"np.multiply by 3: {np.multiply(a, 3).magnitude}")

# --- Comparisons ---

threshold = ArrayQuantity(np.array([2.5, 2.5, 2.5, 2.5, 2.5]), "kilometer", ureg)
mask = distances > threshold
print(f"\nDistances > 2.5 km: {mask}")

# --- Indexing ---

print(f"\nFirst distance: {distances[0]}")
print(f"Last three: {distances[2:].magnitude}")

# --- Reductions (via numpy) ---

print(f"\nSum: {np.sum(distances.magnitude)} {distances.units}")
print(f"Mean: {np.mean(distances.magnitude)} {distances.units}")
print(f"Max: {np.max(distances.magnitude)} {distances.units}")
print(f"Min: {np.min(distances.magnitude)} {distances.units}")

# --- 2D arrays ---

matrix = ArrayQuantity(
    np.array([[1.0, 2.0], [3.0, 4.0]]),
    "meter",
    ureg,
)
print(f"\n2D shape: {matrix.shape}")
print(f"Transpose:\n{matrix.T.magnitude}")
