"""Basic pintrs usage: quantities, conversions, and arithmetic."""

from __future__ import annotations

from pintrs import Measurement, Quantity, UnitRegistry

ureg = UnitRegistry()

# --- Creating quantities ---

distance = ureg.Quantity(100.0, "kilometer")
time = ureg.Quantity(1.5, "hour")
print(f"Distance: {distance}")
print(f"Time: {time}")

# From string expression
q = Quantity("9.81 m/s**2")
print(f"Gravity: {q}")

# Attribute-style
print(f"ureg.meter: {ureg.meter}")
print(f"ureg.speed_of_light: {ureg.speed_of_light}")

# --- Conversions ---

print(f"\n{distance} = {distance.to('mile')}")
print(f"{distance} = {distance.to('meter')}")
print(f"{time} = {time.to('minute')}")

# In-place conversion
speed = distance / time
print(f"\nSpeed: {speed}")
print(f"Speed in m/s: {speed.to('m/s')}")
print(f"Speed compact: {speed.to_compact()}")

# --- Arithmetic ---

d1 = ureg.Quantity(10.0, "meter")
d2 = ureg.Quantity(5.0, "meter")
print(f"\n{d1} + {d2} = {d1 + d2}")
print(f"{d1} - {d2} = {d1 - d2}")
print(f"{d1} * 3 = {d1 * 3}")
print(f"{d1} / 2 = {d1 / 2}")
print(f"{d1} ** 2 = {d1**2}")

# Mixed units
d3 = ureg.Quantity(1.0, "kilometer")
print(f"\n{d1} + {d3} = {d1 + d3}")

# Unit multiplication
area = d1 * ureg.Quantity(4.0, "meter")
print(f"Area: {area}")
print(f"Area in base units: {area.to_base_units()}")

# --- Comparisons ---

print(f"\n{d1} > {d2}: {d1 > d2}")
print(f"{d1} == {d2}: {d1 == d2}")

# --- Properties ---

print(f"\nDimensionality of {speed}: {speed.dimensionality}")
print(f"Is {speed} dimensionless? {speed.dimensionless}")
print(f"Compatible units for meter: {ureg.Quantity(1, 'meter').compatible_units()[:5]}")

# --- Measurements (uncertainty) ---

m = Measurement(Quantity(100.0, "meter"), 0.5)
print(f"\nMeasurement: {m}")
print(f"Relative error: {m.rel:.4f}")

m2 = Measurement(Quantity(50.0, "meter"), 0.3)
combined = m + m2
print(f"{m} + {m2} = {combined}")

# --- Custom units ---

ureg.define("smoot = 1.7018 * meter")
bridge = ureg.Quantity(364.4, "smoot")
print(f"\nHarvard Bridge: {bridge}")
print(f"In meters: {bridge.to('meter')}")
