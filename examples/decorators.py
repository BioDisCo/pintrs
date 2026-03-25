"""Using pintrs decorators for automatic unit handling."""

from __future__ import annotations

from pintrs import UnitRegistry, check, wraps

ureg = UnitRegistry()

# --- @wraps: automatic argument conversion and return unit ---


@wraps(ureg, ret="meter/second", args=("meter", "second"))
def compute_speed(distance: float, time: float) -> float:
    """Arguments are auto-converted to base units, result gets units attached."""
    return distance / time


distance = ureg.Quantity(100, "kilometer")
time = ureg.Quantity(2, "hour")
result = compute_speed(distance, time)
print(f"Speed: {result}")
print(f"Speed in km/h: {result.to('km/hr')}")

# --- @check: dimension validation ---


@check(ureg, "[length]", "[time]")
def velocity(d: object, t: object) -> object:
    """Validates that arguments have the right dimensions before calling."""
    return d / t  # type: ignore[operator]


v = velocity(ureg.Quantity(100, "km"), ureg.Quantity(2, "hr"))
print(f"\nVelocity: {v}")

# Passing wrong dimensions raises DimensionalityError
try:
    velocity(ureg.Quantity(100, "kg"), ureg.Quantity(2, "hr"))
except Exception as e:
    print(f"Caught: {e}")

# --- @wraps with None args (strip units, no conversion) ---


@wraps(ureg, ret="joule", args=(None, None))
def kinetic_energy(mass: float, velocity: float) -> float:
    """Compute kinetic energy. Args have units stripped, result gets joules."""
    return 0.5 * mass * velocity**2


ke = kinetic_energy(
    ureg.Quantity(10.0, "kg"),
    ureg.Quantity(5.0, "m/s"),
)
print(f"\nKinetic energy: {ke}")
print(f"In kJ: {ke.to('kJ')}")
