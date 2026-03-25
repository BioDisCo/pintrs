"""Unit formatting support for pintrs.

Provides pretty (Unicode), LaTeX, HTML, and compact formatting modes
compatible with pint's format specifiers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pintrs._core import Quantity, Unit

_SUPERSCRIPT_MAP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def _format_default_magnitude(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _parse_format_spec(spec: str) -> tuple[str, str, bool]:
    """Parse a pint-compatible format spec.

    Returns:
        Tuple of (magnitude_format, unit_mode, use_abbreviations).
        unit_mode is one of "D", "P", "L", "H", "C".
    """
    use_abbrev = "~" in spec
    clean = spec.replace("~", "")

    if clean and clean[-1] in "PLHCD":
        unit_mode = clean[-1]
        mag_fmt = clean[:-1]
    else:
        unit_mode = "D"
        mag_fmt = clean

    return mag_fmt, unit_mode, use_abbrev


def _get_components(unit: Unit, use_abbrev: bool) -> list[tuple[str, float]]:
    """Get (display_name, exponent) pairs from a Unit.

    With use_abbrev=True, names come from _units_str (symbols).
    With use_abbrev=False, names come from _units_dict (canonical names).
    """
    raw = list(unit._units_dict())
    if not use_abbrev:
        return raw
    # Build a symbol mapping from the _units_str output
    symbol_str = unit._units_str()
    return _parse_unit_str_to_components(symbol_str, raw)


def _parse_unit_str_to_components(
    unit_str: str,
    raw: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    """Parse _units_str output to extract symbol names with exponents.

    Falls back to raw components if parsing fails.
    """
    if not unit_str or unit_str == "dimensionless":
        return raw

    result: list[tuple[str, float]] = []
    # Split on / to get numerator and denominator parts
    parts = unit_str.split(" / ")
    numer_tokens = parts[0].split(" * ") if parts[0] != "1" else []
    denom_tokens = parts[1:] if len(parts) > 1 else []

    for token in numer_tokens:
        name, exp = _parse_unit_token(token)
        result.append((name, exp))
    for token in denom_tokens:
        name, exp = _parse_unit_token(token)
        result.append((name, -exp))

    return result if result else raw


def _parse_unit_token(token: str) -> tuple[str, float]:
    """Parse a single unit token like 's ** 2' into ('s', 2.0)."""
    token = token.strip()
    if " ** " in token:
        name, exp_str = token.rsplit(" ** ", 1)
        return name.strip(), float(exp_str)
    return token, 1.0


def _format_components(
    components: list[tuple[str, float]],
    *,
    exp_formatter: str = "default",
    separator: str = " * ",
    div_char: str = " / ",
) -> str:
    """Generic component formatter.

    Args:
        components: (name, exponent) pairs.
        exp_formatter: "default" for ** N, "pretty" for Unicode,
            "latex" for ^{N}, "html" for <sup>N</sup>.
        separator: Separator between units in numerator/denominator.
        div_char: Separator between numerator and denominator.
    """
    pos = [(n, e) for n, e in components if e > 0]
    neg = [(n, e) for n, e in components if e < 0]
    pos.sort(key=lambda x: x[0])
    neg.sort(key=lambda x: x[0])

    _exp_epsilon = 1e-9

    def _fmt(name: str, exp: float) -> str:
        a = abs(exp)
        if abs(a - 1.0) < _exp_epsilon:
            return name
        if exp_formatter == "pretty":
            s = str(int(a)) if a == int(a) else str(a)
            return name + s.translate(_SUPERSCRIPT_MAP)
        if exp_formatter == "latex":
            escaped = name.replace("_", r"\_")
            e_str = str(int(a)) if a == int(a) else str(a)
            return rf"\mathrm{{{escaped}}}^{{{e_str}}}"
        if exp_formatter == "html":
            e_str = str(int(a)) if a == int(a) else str(a)
            return f"{name}<sup>{e_str}</sup>"
        # default
        e_str = str(int(a)) if a == int(a) else str(a)
        return f"{name} ** {e_str}"

    if exp_formatter == "latex":
        pos_parts = [_fmt(n, e) for n, e in pos]
        neg_parts = [_fmt(n, abs(e)) for n, e in neg]
        numer = r" \cdot ".join(pos_parts) if pos_parts else "1"
        if not neg_parts:
            return rf"${numer}$"
        denom = r" \cdot ".join(neg_parts)
        return rf"$\frac{{{numer}}}{{{denom}}}$"

    pos_parts = [_fmt(n, e) for n, e in pos]
    neg_parts = [_fmt(n, abs(e)) for n, e in neg]

    if not neg_parts:
        return separator.join(pos_parts) if pos_parts else "dimensionless"
    if not pos_parts:
        return "1" + div_char + div_char.join(neg_parts)
    return div_char.join([separator.join(pos_parts), *neg_parts])


def format_unit(unit: Unit, spec: str = "") -> str:
    """Format a Unit with pint-compatible format specifiers.

    Args:
        unit: The unit to format.
        spec: Format specifier. Supports ~, P, L, H, C, D suffixes.
    """
    _, unit_mode, use_abbrev = _parse_format_spec(spec)
    components = _get_components(unit, use_abbrev)

    if unit_mode == "P":
        return _format_components(
            components,
            exp_formatter="pretty",
            separator="·",
            div_char="/",
        )
    if unit_mode == "L":
        return _format_components(components, exp_formatter="latex")
    if unit_mode == "H":
        return _format_components(
            components,
            exp_formatter="html",
            separator=" ",
            div_char="/",
        )
    # D or C
    return _format_components(components)


def format_quantity(quantity: Quantity, spec: str = "") -> str:
    """Format a Quantity with pint-compatible format specifiers.

    Args:
        quantity: The quantity to format.
        spec: Format specifier like ".2f~P", "~L", ".3g~H", etc.
    """
    mag_fmt, unit_mode, use_abbrev = _parse_format_spec(spec)

    q = quantity
    if unit_mode == "C":
        q = quantity.to_compact()

    mag_str = (
        format(q.magnitude, mag_fmt)
        if mag_fmt
        else _format_default_magnitude(
            q.magnitude,
        )
    )

    unit_spec = ("~" if use_abbrev else "") + unit_mode
    unit_str = format_unit(q.units, unit_spec)

    return f"{mag_str} {unit_str}"
