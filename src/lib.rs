mod array_quantity;
mod definition;
mod errors;
mod parser;
mod registry;
mod units_container;

use std::sync::{Arc, Mutex};

use once_cell::sync::OnceCell;
use pyo3::exceptions::{PyTypeError, PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;

use errors::PintError;
use registry::UnitRegistry as RustRegistry;
use units_container::UnitsContainer;

pub(crate) type SharedRegistry = Arc<Mutex<RustRegistry>>;

// --- Custom exception types ---

pyo3::create_exception!(
    _core,
    PintError_Py,
    pyo3::exceptions::PyValueError,
    "Base pint error."
);
pyo3::create_exception!(
    _core,
    DimensionalityError,
    PintError_Py,
    "Raised when converting between incompatible dimensions."
);
pyo3::create_exception!(
    _core,
    UndefinedUnitError,
    PintError_Py,
    "Raised when a unit is not found in the registry."
);
pyo3::create_exception!(
    _core,
    OffsetUnitCalculusError,
    PintError_Py,
    "Raised on ambiguous operations with offset units."
);
pyo3::create_exception!(
    _core,
    DefinitionSyntaxError,
    PintError_Py,
    "Raised on malformed unit definitions."
);
pyo3::create_exception!(
    _core,
    RedefinitionError,
    PintError_Py,
    "Raised when redefining an existing unit."
);

pub(crate) fn to_py_err(e: Box<PintError>) -> PyErr {
    match *e {
        PintError::DimensionalityError { .. } => DimensionalityError::new_err(e.to_string()),
        PintError::UndefinedUnitError { .. } => UndefinedUnitError::new_err(e.to_string()),
        PintError::OffsetUnitCalculusError { .. } => {
            OffsetUnitCalculusError::new_err(e.to_string())
        }
        PintError::DefinitionSyntaxError { .. } => DefinitionSyntaxError::new_err(e.to_string()),
        PintError::RedefinitionError { .. } => RedefinitionError::new_err(e.to_string()),
    }
}

/// Convert a registry `PintResult` into a `PyResult`, mapping the domain error
/// to its Python exception. Lets call sites write `reg.foo(..).to_py()?` instead
/// of the noisier `reg.foo(..).map_err(to_py_err)?`.
pub(crate) trait PintResultExt<T> {
    fn to_py(self) -> PyResult<T>;
}

impl<T> PintResultExt<T> for errors::PintResult<T> {
    #[inline]
    fn to_py(self) -> PyResult<T> {
        self.map_err(to_py_err)
    }
}

// --- UnitRegistry ---

#[pyclass(name = "UnitRegistry", module = "pintrs._core", subclass)]
pub(crate) struct PyUnitRegistry {
    pub(crate) inner: SharedRegistry,
    #[pyo3(get)]
    _init_kwargs: Option<PyObject>,
}

#[pymethods]
impl PyUnitRegistry {
    #[new]
    #[pyo3(signature = (**kwargs))]
    fn new(_py: Python<'_>, kwargs: Option<Bound<'_, pyo3::types::PyDict>>) -> PyResult<Self> {
        let reg = RustRegistry::new().to_py()?;
        let stored: Option<PyObject> = kwargs.map(|d| d.unbind().into_any());
        Ok(Self {
            inner: Arc::new(Mutex::new(reg)),
            _init_kwargs: stored,
        })
    }

    /// Fast path: f64 magnitude + unit string, no type dispatch needed.
    #[pyo3(name = "_f64_quantity")]
    #[pyo3(signature = (value, units=None))]
    #[inline]
    fn f64_quantity(
        slf: &Bound<'_, Self>,
        value: f64,
        units: Option<&str>,
    ) -> PyResult<PyQuantity> {
        let this = slf.borrow();
        let units_str = units.unwrap_or("dimensionless");
        let mut reg = this.inner.lock().unwrap();
        let uc = reg.parse_unit_expr(units_str).to_py()?;
        let cached = OnceCell::new();
        let _ = cached.set(slf.as_any().clone().unbind());
        Ok(PyQuantity {
            magnitude: value,
            units: uc,
            registry: Arc::clone(&this.inner),
            cached_dim: OnceCell::new(),
            cached_registry_obj: cached,
        })
    }

    #[pyo3(name = "_scalar_quantity")]
    #[pyo3(signature = (value, units=None))]
    fn quantity(
        slf: &Bound<'_, Self>,
        value: &Bound<'_, PyAny>,
        units: Option<&str>,
    ) -> PyResult<PyQuantity> {
        let this = slf.borrow();
        let reg_obj = slf.as_any().clone().unbind();

        // Quantity(Quantity, new_units) -> convert
        if let Ok(other_q) = value.extract::<PyQuantity>() {
            return match units {
                Some(u) => {
                    let mut reg = this.inner.lock().unwrap();
                    let dst = reg.parse_unit_expr(u).to_py()?;
                    let new_mag = reg
                        .convert(other_q.magnitude, &other_q.units, &dst)
                        .to_py()?;
                    let cached = OnceCell::new();
                    let _ = cached.set(reg_obj);
                    Ok(PyQuantity {
                        magnitude: new_mag,
                        units: dst,
                        registry: Arc::clone(&this.inner),
                        cached_dim: OnceCell::new(),
                        cached_registry_obj: cached,
                    })
                }
                None => {
                    let cached = OnceCell::new();
                    let _ = cached.set(reg_obj);
                    Ok(PyQuantity {
                        magnitude: other_q.magnitude,
                        units: other_q.units.clone(),
                        registry: Arc::clone(&this.inner),
                        cached_dim: OnceCell::new(),
                        cached_registry_obj: cached,
                    })
                }
            };
        }

        // Quantity(string) -> parse
        if let Ok(s) = value.extract::<String>() {
            if units.is_some() {
                return Err(PyValueError::new_err(
                    "Cannot specify units when value is a string",
                ));
            }
            return this.parse_expression(&s);
        }

        // Quantity(number, units)
        let mag: f64 = value.extract()?;
        let units_str = units.unwrap_or("dimensionless");
        let mut reg = this.inner.lock().unwrap();
        let uc = reg.parse_unit_expr(units_str).to_py()?;
        let cached = OnceCell::new();
        let _ = cached.set(reg_obj);
        Ok(PyQuantity {
            magnitude: mag,
            units: uc,
            registry: Arc::clone(&this.inner),
            cached_dim: OnceCell::new(),
            cached_registry_obj: cached,
        })
    }

    #[pyo3(name = "Unit")]
    fn unit(&self, units: &str) -> PyResult<PyUnit> {
        let mut reg = self.inner.lock().unwrap();
        let uc = reg.parse_unit_expr(units).to_py()?;
        Ok(PyUnit {
            units: uc,
            registry: Arc::clone(&self.inner),
            cached_registry_obj: OnceCell::new(),
        })
    }

    #[pyo3(name = "parse_expression")]
    fn parse_expression(&self, expr: &str) -> PyResult<PyQuantity> {
        let expr = expr.trim();
        if expr.is_empty() {
            return Err(PyValueError::new_err("Cannot parse empty expression"));
        }
        let (mag, unit_str) = parse_quantity_string(expr);
        let magnitude: f64 = mag
            .parse()
            .map_err(|_| PyValueError::new_err(format!("Cannot parse magnitude: '{}'", mag)))?;
        let mut reg = self.inner.lock().unwrap();
        let units = if unit_str.is_empty() {
            UnitsContainer::new()
        } else {
            reg.parse_unit_expr(unit_str).to_py()?
        };
        Ok(PyQuantity {
            magnitude,
            units,
            registry: Arc::clone(&self.inner),
            cached_dim: OnceCell::new(),
            cached_registry_obj: OnceCell::new(),
        })
    }

    #[pyo3(name = "parse_units")]
    fn parse_units(&self, units: &str) -> PyResult<PyUnit> {
        self.unit(units)
    }

    #[pyo3(name = "convert")]
    fn convert_val(&self, value: f64, src: &str, dst: &str) -> PyResult<f64> {
        let mut reg = self.inner.lock().unwrap();
        let src_uc = reg.parse_unit_expr(src).to_py()?;
        let dst_uc = reg.parse_unit_expr(dst).to_py()?;
        reg.convert(value, &src_uc, &dst_uc).to_py()
    }

    /// Define a new unit at runtime: ureg.define("smoot = 1.7018 * meter")
    fn define(&self, definition: &str) -> PyResult<()> {
        let defs = crate::parser::parse_definitions(definition);
        let mut reg = self.inner.lock().unwrap();
        for def in defs {
            match def {
                crate::definition::Definition::Unit(u) => {
                    reg.define_unit(&u).to_py()?;
                }
                crate::definition::Definition::Prefix(p) => {
                    reg.define_prefix(&p);
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Get compatible units for a given unit
    fn get_compatible_units(&self, unit: &str) -> PyResult<Vec<String>> {
        let mut reg = self.inner.lock().unwrap();
        let uc = reg.parse_unit_expr(unit).to_py()?;
        let dim = reg.get_dimensionality(&uc).to_py()?;
        Ok(reg.get_units_by_dimensionality(&dim))
    }

    /// Internal: whether a unit expression is non-multiplicative (offset or
    /// logarithmic), so the Python array wrapper applies the proper transform.
    fn _is_non_multiplicative(&self, unit: &str) -> PyResult<bool> {
        let mut reg = self.inner.lock().unwrap();
        let uc = reg.parse_unit_expr(unit).to_py()?;
        Ok(reg.is_non_multiplicative(&uc))
    }

    /// Whether `unit` involves a logarithmic unit (e.g. `dB`, `dBW`). Used by
    /// the Python layer to classify offset (affine) vs logarithmic conversions
    /// when applying them to complex magnitudes.
    fn _is_log_unit(&self, unit: &str) -> PyResult<bool> {
        let mut reg = self.inner.lock().unwrap();
        let uc = reg.parse_unit_expr(unit).to_py()?;
        Ok(reg.has_log_units(&uc))
    }

    /// Internal: get conversion factor (used by Python wrapper)
    fn _get_conversion_factor(&self, src: &str, dst: &str) -> PyResult<f64> {
        let mut reg = self.inner.lock().unwrap();
        let src_uc = reg.parse_unit_expr(src).to_py()?;
        let dst_uc = reg.parse_unit_expr(dst).to_py()?;
        reg.get_conversion_factor(&src_uc, &dst_uc).to_py()
    }

    /// Internal: get root units representation
    fn _get_root_units(&self, unit: &str) -> PyResult<(f64, String)> {
        let mut reg = self.inner.lock().unwrap();
        let uc = reg.parse_unit_expr(unit).to_py()?;
        let (factor, root) = reg.get_root_units(&uc).to_py()?;
        Ok((factor, reg.format_units(&root)))
    }

    /// Internal: get dimensionality string
    fn _get_dimensionality(&self, unit: &str) -> PyResult<String> {
        let mut reg = self.inner.lock().unwrap();
        let uc = reg.parse_unit_expr(unit).to_py()?;
        let dim = reg.get_dimensionality(&uc).to_py()?;
        Ok(dim.to_string())
    }

    /// Internal: format units
    fn _format_units(&self, unit: &str) -> PyResult<String> {
        let mut reg = self.inner.lock().unwrap();
        let uc = reg.parse_unit_expr(unit).to_py()?;
        Ok(reg.format_units(&uc))
    }

    /// Internal: list all known prefixes with their factors
    fn _get_prefixes(&self) -> Vec<(String, f64)> {
        let reg = self.inner.lock().unwrap();
        reg.get_all_prefixes()
    }

    /// Get the base units for a unit expression. Returns (factor, Unit).
    fn get_base_units(&self, unit: &str) -> PyResult<(f64, PyUnit)> {
        let mut reg = self.inner.lock().unwrap();
        let uc = reg.parse_unit_expr(unit).to_py()?;
        let (factor, root) = reg.get_root_units(&uc).to_py()?;
        Ok((
            factor,
            PyUnit {
                units: root,
                registry: Arc::clone(&self.inner),
                cached_registry_obj: OnceCell::new(),
            },
        ))
    }

    /// Get root units (alias for get_base_units).
    fn get_root_units(&self, unit: &str) -> PyResult<(f64, PyUnit)> {
        self.get_base_units(unit)
    }

    /// Get the dimensionality of a unit expression.
    fn get_dimensionality(&self, unit: &str) -> PyResult<String> {
        self._get_dimensionality(unit)
    }

    /// Get the canonical name for a unit.
    fn get_name(&self, unit: &str) -> PyResult<String> {
        let reg = self.inner.lock().unwrap();
        reg.get_canonical_name(unit).to_py()
    }

    /// Get the symbol for a unit.
    fn get_symbol(&self, unit: &str) -> PyResult<String> {
        let reg = self.inner.lock().unwrap();
        let canonical = reg.get_canonical_name(unit).to_py()?;
        Ok(reg.get_display_name(&canonical, true))
    }

    /// Check if two unit expressions are dimensionally compatible.
    fn is_compatible_with(&self, unit1: &str, unit2: &str) -> PyResult<bool> {
        let mut reg = self.inner.lock().unwrap();
        let uc1 = reg.parse_unit_expr(unit1).to_py()?;
        let uc2 = reg.parse_unit_expr(unit2).to_py()?;
        let dim1 = reg.get_dimensionality(&uc1).to_py()?;
        let dim2 = reg.get_dimensionality(&uc2).to_py()?;
        Ok(dim1 == dim2)
    }

    /// Parse a unit name into (prefix, unit_name, suffix) tuples.
    fn parse_unit_name(&self, name: &str) -> PyResult<Vec<(String, String, String)>> {
        let reg = self.inner.lock().unwrap();
        let _canonical = match reg.get_canonical_name(name) {
            Ok(c) => c,
            Err(e) => return Err(to_py_err(e)),
        };
        // Try to detect prefix
        let result = reg.parse_prefix_and_unit(name);
        Ok(result)
    }

    /// Load additional definitions from a string.
    fn load_definitions(&self, text: &str) -> PyResult<()> {
        let defs = crate::parser::parse_definitions(text);
        let mut reg = self.inner.lock().unwrap();
        for def in defs {
            match def {
                crate::definition::Definition::Unit(u) => {
                    let _ = reg.define_unit(&u);
                }
                crate::definition::Definition::Prefix(p) => {
                    reg.define_prefix(&p);
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Stub for default_format property.
    #[getter]
    fn default_format(&self) -> String {
        "D".to_string()
    }

    /// Stub for auto_reduce_dimensions property.
    #[getter]
    fn auto_reduce_dimensions(&self) -> bool {
        false
    }

    /// Check if a unit is known
    fn __contains__(&self, item: &str) -> bool {
        let reg = self.inner.lock().unwrap();
        reg.is_known_unit(item)
    }

    fn __iter__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let reg = self.inner.lock().unwrap();
        let mut names: Vec<String> = reg.get_all_unit_names();
        names.sort();
        let list = pyo3::types::PyList::new(py, names)?;
        Ok(list.call_method0("__iter__")?.unbind())
    }

    fn __getattr__(slf: &Bound<'_, Self>, name: &str) -> PyResult<PyUnit> {
        if name.starts_with('_') {
            return Err(pyo3::exceptions::PyAttributeError::new_err(format!(
                "'UnitRegistry' object has no attribute '{}'",
                name
            )));
        }
        let this = slf.borrow();
        let mut reg = this.inner.lock().unwrap();
        if !reg.is_known_unit(name) {
            return Err(pyo3::exceptions::PyAttributeError::new_err(format!(
                "'{}' is not defined in the unit registry",
                name
            )));
        }
        let uc = reg.parse_unit_expr(name).to_py()?;
        let cached = OnceCell::new();
        let _ = cached.set(slf.as_any().clone().unbind());
        Ok(PyUnit {
            units: uc,
            registry: Arc::clone(&this.inner),
            cached_registry_obj: cached,
        })
    }

    fn __reduce__(&self, py: Python<'_>) -> PyResult<(PyObject, (Option<PyObject>,))> {
        let cls = py.get_type::<PyUnitRegistry>();
        let kwargs = self._init_kwargs.as_ref().map(|k| k.clone_ref(py));
        Ok((cls.unbind().into_any(), (kwargs,)))
    }

    fn __copy__(slf: &Bound<'_, Self>) -> PyResult<PyObject> {
        Ok(slf.as_any().clone().unbind())
    }

    fn __deepcopy__(slf: &Bound<'_, Self>, _memo: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        Ok(slf.as_any().clone().unbind())
    }
}

// --- Quantity ---

#[pyclass(name = "Quantity", module = "pintrs._core", subclass, dict)]
pub(crate) struct PyQuantity {
    pub(crate) magnitude: f64,
    pub(crate) units: UnitsContainer,
    pub(crate) registry: SharedRegistry,
    pub(crate) cached_dim: OnceCell<String>,
    /// Cached Python-level registry wrapper (for `_REGISTRY is ureg` identity).
    pub(crate) cached_registry_obj: OnceCell<PyObject>,
}

impl Clone for PyQuantity {
    fn clone(&self) -> Self {
        Self {
            magnitude: self.magnitude,
            units: self.units.clone(),
            registry: Arc::clone(&self.registry),
            cached_dim: self.cached_dim.clone(),
            cached_registry_obj: OnceCell::new(), // don't clone the cached PyObject
        }
    }
}

impl PyQuantity {
    /// Build a `Quantity` from its parts, with both caches empty. Centralises
    /// the construction boilerplate shared by every arithmetic operator.
    #[inline]
    pub(crate) fn from_parts(
        magnitude: f64,
        units: UnitsContainer,
        registry: SharedRegistry,
    ) -> Self {
        Self {
            magnitude,
            units,
            registry,
            cached_dim: OnceCell::new(),
            cached_registry_obj: OnceCell::new(),
        }
    }

    /// Build a new `Quantity` sharing this one's registry. The common case for
    /// operators that produce a result in the same registry as `self`.
    #[inline]
    fn rebuild(&self, magnitude: f64, units: UnitsContainer) -> Self {
        Self::from_parts(magnitude, units, Arc::clone(&self.registry))
    }

    /// Adopt the magnitude and units of `q` in place. Lets each `ito_*` (the
    /// in-place conversions) delegate to its `to_*` counterpart instead of
    /// re-implementing the conversion.
    #[inline]
    fn assign_from(&mut self, q: PyQuantity) {
        self.magnitude = q.magnitude;
        self.units = q.units;
    }

    /// `self`'s value expressed as a plain (empty) dimensionless number.
    ///
    /// Used by operations that mix in a bare Python number (floor-divide, modulo,
    /// ordered comparisons, dimensionless add/sub): pint requires the quantity to
    /// be dimensionless and operates on its value in base units, not the
    /// intermediate (scaled) units such as `cm*s/m/ms`. The conversion is
    /// offset/log-aware, so a logarithmic-but-dimensionless unit like `dB` yields
    /// its linear value (`1 dB -> 10**0.1`), matching pint. Returns
    /// `DimensionalityError` if `self` is not dimensionless.
    fn dimensionless_value(&self) -> PyResult<f64> {
        let mut reg = self.registry.lock().unwrap();
        let dim = reg.get_dimensionality(&self.units).to_py()?;
        if !dim.is_empty() {
            return Err(DimensionalityError::new_err(
                "Cannot operate with a dimensioned quantity and a bare number",
            ));
        }
        reg.convert(self.magnitude, &self.units, &UnitsContainer::new())
            .to_py()
    }

    /// Multiplicative factor from `self.units` to plain dimensionless units.
    /// Like [`dimensionless_value`](Self::dimensionless_value) but returns the
    /// scale, for the one caller (`quantity % number`) that must convert a
    /// base-unit result back into `self.units`. Multiplicative only.
    fn require_dimensionless_factor(&self) -> PyResult<f64> {
        let mut reg = self.registry.lock().unwrap();
        let dim = reg.get_dimensionality(&self.units).to_py()?;
        if !dim.is_empty() {
            return Err(DimensionalityError::new_err(
                "Cannot operate with a dimensioned quantity and a bare number",
            ));
        }
        reg.get_conversion_factor(&self.units, &UnitsContainer::new())
            .to_py()
    }

    /// Raise `OffsetUnitCalculusError` if `units` is non-multiplicative (an
    /// offset unit such as `degC` or a logarithmic unit such as `dB`). pint
    /// forbids these from taking part in multiplication, division, or a power
    /// other than 0 or 1. Pass `&self.units` to guard the receiver, or an
    /// operand's units to guard the other side. `delta_*` units are
    /// multiplicative and pass.
    fn ensure_multiplicative(&self, units: &UnitsContainer) -> PyResult<()> {
        let reg = self.registry.lock().unwrap();
        ensure_units_multiplicative(&reg, units)
    }

    /// Shared body of the ordered comparisons (`<`, `<=`, `>`, `>=`). Another
    /// quantity is compared in `self`'s units (offset/log-aware via `convert`),
    /// and a bare number requires `self` to be dimensionless and uses its base
    /// value. `cmp` is the concrete operator.
    fn compare_ordered(
        &self,
        other: &Bound<'_, PyAny>,
        cmp: fn(f64, f64) -> bool,
    ) -> PyResult<bool> {
        if let Ok(other_q) = other.extract::<PyQuantity>() {
            if self.units == other_q.units {
                return Ok(cmp(self.magnitude, other_q.magnitude));
            }
            let mut reg = self.registry.lock().unwrap();
            let converted = reg
                .convert(other_q.magnitude, &other_q.units, &self.units)
                .to_py()?;
            Ok(cmp(self.magnitude, converted))
        } else if let Ok(val) = other.extract::<f64>() {
            if val == 0.0 {
                return Ok(cmp(self.magnitude, 0.0));
            }
            let value = self.dimensionless_value()?;
            Ok(cmp(value, val))
        } else {
            Err(PyTypeError::new_err(
                "Cannot compare Quantity with non-Quantity",
            ))
        }
    }
}

/// Raise `OffsetUnitCalculusError` if `units` is non-multiplicative (offset or
/// logarithmic). Shared across `Quantity`, `Unit` and array arithmetic so that
/// an offset operand on either side of `*`/`/`/`**` is rejected the way pint
/// does.
pub(crate) fn ensure_units_multiplicative(
    reg: &RustRegistry,
    units: &UnitsContainer,
) -> PyResult<()> {
    if reg.is_non_multiplicative(units) {
        return Err(OffsetUnitCalculusError::new_err(format!(
            "Ambiguous operation with offset unit ({units}, dimensionless)"
        )));
    }
    Ok(())
}

/// Return (and memoise) the Python `UnitRegistry` wrapper for a shared registry.
/// Shared by the `_registry` getter on `Quantity`, `Unit` and `RustArrayQuantity`,
/// which all cache the wrapper in a `OnceCell` to avoid rebuilding it.
pub(crate) fn registry_pyobject(
    py: Python<'_>,
    registry: &SharedRegistry,
    cache: &OnceCell<PyObject>,
) -> PyResult<PyObject> {
    if let Some(cached) = cache.get() {
        return Ok(cached.clone_ref(py));
    }
    let reg = PyUnitRegistry {
        inner: Arc::clone(registry),
        _init_kwargs: None,
    };
    let obj = reg.into_pyobject(py)?.into_any().unbind();
    let _ = cache.set(obj.clone_ref(py));
    Ok(obj)
}

/// Floating-point `divmod` matching CPython's `float.__divmod__` exactly, so
/// `//` and `%` agree with Python (and pint) for negative operands, infinities
/// and signed zeros. Returns `(floordiv, mod)`.
///
/// Unlike Rust's `%` / `fmod` (sign of the dividend), the remainder takes the
/// sign of the divisor; infinite divisors and `inf // finite` are handled the
/// way CPython does rather than producing a stray `inf`/`NaN`.
fn py_divmod(a: f64, b: f64) -> (f64, f64) {
    let mut m = a % b;
    let mut d = (a - m) / b;
    if m != 0.0 {
        if (b < 0.0) != (m < 0.0) {
            m += b;
            d -= 1.0;
        }
    } else {
        m = 0.0_f64.copysign(b);
    }
    let floordiv = if d != 0.0 {
        let fl = d.floor();
        if d - fl > 0.5 {
            fl + 1.0
        } else {
            fl
        }
    } else {
        0.0_f64.copysign(a / b)
    };
    (floordiv, m)
}

/// Python/pint-style floored modulo (sign of the divisor).
#[inline]
fn floored_mod(a: f64, b: f64) -> f64 {
    py_divmod(a, b).1
}

/// Python/pint-style floor division.
#[inline]
fn floored_div(a: f64, b: f64) -> f64 {
    py_divmod(a, b).0
}

#[pymethods]
impl PyQuantity {
    #[new]
    #[pyo3(signature = (value, units=None))]
    fn new(value: &Bound<'_, PyAny>, units: Option<&str>) -> PyResult<Self> {
        let reg = RustRegistry::new().to_py()?;
        let shared = Arc::new(Mutex::new(reg));

        // Quantity(Quantity, new_units)
        if let Ok(other_q) = value.extract::<PyQuantity>() {
            return match units {
                Some(u) => {
                    let (new_mag, dst) = {
                        let mut r = shared.lock().unwrap();
                        let dst = r.parse_unit_expr(u).to_py()?;
                        let new_mag = r.convert(other_q.magnitude, &other_q.units, &dst).to_py()?;
                        (new_mag, dst)
                    };
                    Ok(Self {
                        magnitude: new_mag,
                        units: dst,
                        registry: shared,
                        cached_dim: OnceCell::new(),
                        cached_registry_obj: OnceCell::new(),
                    })
                }
                None => Ok(Self {
                    magnitude: other_q.magnitude,
                    units: other_q.units.clone(),
                    registry: shared,
                    cached_dim: OnceCell::new(),
                    cached_registry_obj: OnceCell::new(),
                }),
            };
        }

        // Quantity(string)
        if let Ok(s) = value.extract::<String>() {
            if units.is_some() {
                return Err(PyValueError::new_err(
                    "Cannot specify units when value is a string",
                ));
            }
            let (mag, unit_str) = parse_quantity_string(&s);
            let magnitude: f64 = mag
                .parse()
                .map_err(|_| PyValueError::new_err(format!("Cannot parse magnitude: '{}'", mag)))?;
            let uc = {
                let mut r = shared.lock().unwrap();
                r.parse_unit_expr(unit_str).to_py()?
            };
            return Ok(Self {
                magnitude,
                units: uc,
                registry: shared,
                cached_dim: OnceCell::new(),
                cached_registry_obj: OnceCell::new(),
            });
        }

        // Quantity(number, units)
        let mag: f64 = value.extract()?;
        let units_str = units.unwrap_or("dimensionless");
        let uc = {
            let mut r = shared.lock().unwrap();
            r.parse_unit_expr(units_str).to_py()?
        };
        Ok(Self {
            magnitude: mag,
            units: uc,
            registry: shared,
            cached_dim: OnceCell::new(),
            cached_registry_obj: OnceCell::new(),
        })
    }

    #[getter]
    fn magnitude(&self) -> f64 {
        self.magnitude
    }

    #[getter]
    fn m(&self) -> f64 {
        self.magnitude()
    }

    #[getter]
    fn _registry(&self, py: Python<'_>) -> PyResult<PyObject> {
        registry_pyobject(py, &self.registry, &self.cached_registry_obj)
    }

    #[getter]
    fn units(&self) -> PyUnit {
        PyUnit {
            units: self.units.clone(),
            registry: Arc::clone(&self.registry),
            cached_registry_obj: OnceCell::new(),
        }
    }

    #[getter]
    fn u(&self) -> PyUnit {
        self.units()
    }

    fn m_as(&self, units: &Bound<'_, PyAny>) -> PyResult<f64> {
        let units_str = extract_units_string(units)?;
        self._to(&units_str).map(|q| q.magnitude)
    }

    fn to(&self, units: &Bound<'_, PyAny>) -> PyResult<PyQuantity> {
        let units_str = extract_units_string(units)?;
        self._to(&units_str)
    }

    fn _to(&self, units: &str) -> PyResult<PyQuantity> {
        let mut reg = self.registry.lock().unwrap();
        let dst = reg.parse_unit_expr(units).to_py()?;
        let new_mag = reg.convert(self.magnitude, &self.units, &dst).to_py()?;
        Ok(self.rebuild(new_mag, dst))
    }

    fn ito(&mut self, units: &Bound<'_, PyAny>) -> PyResult<()> {
        let units_str = extract_units_string(units)?;
        let q = self._to(&units_str)?;
        self.assign_from(q);
        Ok(())
    }

    fn to_base_units(&self) -> PyResult<PyQuantity> {
        let mut reg = self.registry.lock().unwrap();
        let (factor, root_units) = reg.get_root_units(&self.units).to_py()?;
        // Offset (non-multiplicative) units like degC must apply their offset when
        // converting to base units, not just the multiplicative factor.
        let magnitude = if reg.has_offset_units(&self.units) || reg.has_log_units(&self.units) {
            reg.convert(self.magnitude, &self.units, &root_units)
                .to_py()?
        } else {
            self.magnitude * factor
        };
        Ok(self.rebuild(magnitude, root_units))
    }

    fn to_root_units(&self) -> PyResult<PyQuantity> {
        self.to_base_units()
    }

    fn ito_base_units(&mut self) -> PyResult<()> {
        let q = self.to_base_units()?;
        self.assign_from(q);
        Ok(())
    }

    fn ito_root_units(&mut self) -> PyResult<()> {
        self.ito_base_units()
    }

    /// Convert to "compact" form with the most appropriate SI prefix.
    #[pyo3(signature = (unit=None))]
    fn to_compact(&self, unit: Option<&str>) -> PyResult<PyQuantity> {
        let mut reg = self.registry.lock().unwrap();

        // If a target unit is given, convert to that first, then compact
        let (mag, units) = if let Some(u) = unit {
            let dst = reg.parse_unit_expr(u).to_py()?;
            let m = reg.convert(self.magnitude, &self.units, &dst).to_py()?;
            (m, dst)
        } else {
            (self.magnitude, self.units.clone())
        };

        // Only compact single-unit quantities
        if units.len() != 1 {
            return Ok(self.rebuild(mag, units));
        }

        let (unit_name, &exp) = units.iter().next().unwrap();
        if (exp - 1.0).abs() > f64::EPSILON {
            return Ok(self.rebuild(mag, units));
        }

        // SI decimal prefixes only (exclude binary kibi/mebi/etc. and oddities like semi/sesqui)
        static SI_PREFIXES: &[(&str, f64)] = &[
            ("quecto", 1e-30),
            ("ronto", 1e-27),
            ("yocto", 1e-24),
            ("zepto", 1e-21),
            ("atto", 1e-18),
            ("femto", 1e-15),
            ("pico", 1e-12),
            ("nano", 1e-9),
            ("micro", 1e-6),
            ("milli", 1e-3),
            ("centi", 1e-2),
            ("deci", 1e-1),
            ("deca", 1e1),
            ("hecto", 1e2),
            ("kilo", 1e3),
            ("mega", 1e6),
            ("giga", 1e9),
            ("tera", 1e12),
            ("peta", 1e15),
            ("exa", 1e18),
            ("zetta", 1e21),
            ("yotta", 1e24),
            ("ronna", 1e27),
            ("quetta", 1e30),
        ];

        let abs_mag = mag.abs();
        if abs_mag == 0.0 {
            return Ok(self.rebuild(mag, units));
        }

        let mut best_prefix = "";
        let mut best_factor: Option<f64> = None;

        for &(prefix_name, prefix_factor) in SI_PREFIXES {
            let scaled = abs_mag / prefix_factor;
            if (1.0..1000.0).contains(&scaled) {
                match best_factor {
                    None => {
                        best_prefix = prefix_name;
                        best_factor = Some(prefix_factor);
                    }
                    Some(bf) if prefix_factor > bf => {
                        best_prefix = prefix_name;
                        best_factor = Some(prefix_factor);
                    }
                    _ => {}
                }
            }
        }

        let best_factor = match best_factor {
            Some(f) => f,
            None => return Ok(self.rebuild(mag, units)),
        };

        let new_unit_name = format!("{}{}", best_prefix, unit_name);
        let new_mag = mag / best_factor;
        let new_units = UnitsContainer::from_single(new_unit_name, 1.0);
        Ok(self.rebuild(new_mag, new_units))
    }

    /// Reduce compound units to their simplest form.
    fn to_reduced_units(&self) -> PyResult<PyQuantity> {
        self.to_base_units()
    }

    fn ito_reduced_units(&mut self) -> PyResult<()> {
        self.ito_base_units()
    }

    /// Convert to preferred units (list of unit strings).
    #[pyo3(signature = (preferred_units=None))]
    fn to_preferred(&self, preferred_units: Option<Vec<String>>) -> PyResult<PyQuantity> {
        let mut reg = self.registry.lock().unwrap();
        let preferred = match preferred_units {
            Some(p) if !p.is_empty() => p,
            _ => {
                let (factor, root_units) = reg.get_root_units(&self.units).to_py()?;
                return Ok(self.rebuild(self.magnitude * factor, root_units));
            }
        };
        let self_dim = reg.get_dimensionality(&self.units).to_py()?;

        for pref in &preferred {
            let pref_uc = reg.parse_unit_expr(pref).to_py()?;
            let pref_dim = reg.get_dimensionality(&pref_uc).to_py()?;
            if pref_dim == self_dim {
                let new_mag = reg.convert(self.magnitude, &self.units, &pref_uc).to_py()?;
                return Ok(self.rebuild(new_mag, pref_uc));
            }
        }

        // Fall back to base units
        let (factor, root_units) = reg.get_root_units(&self.units).to_py()?;
        Ok(self.rebuild(self.magnitude * factor, root_units))
    }

    #[pyo3(signature = (preferred_units=None))]
    fn ito_preferred(&mut self, preferred_units: Option<Vec<String>>) -> PyResult<()> {
        let q = self.to_preferred(preferred_units)?;
        self.assign_from(q);
        Ok(())
    }

    /// Convert prefixed units to their unprefixed base form.
    fn to_unprefixed(&self) -> PyResult<PyQuantity> {
        let mut reg = self.registry.lock().unwrap();
        let mut new_units = UnitsContainer::new();
        let mut factor = 1.0_f64;

        for (unit_name, &exp) in self.units.iter() {
            // Try to find the base (unprefixed) unit
            let (uf, root) = reg
                .get_root_units(&UnitsContainer::from_single(unit_name.clone(), 1.0))
                .to_py()?;
            factor *= uf.powf(exp);
            new_units = &new_units * &root.pow(exp);
        }

        Ok(self.rebuild(self.magnitude * factor, new_units))
    }

    fn ito_unprefixed(&mut self) -> PyResult<()> {
        let q = self.to_unprefixed()?;
        self.assign_from(q);
        Ok(())
    }

    /// Return (magnitude, units_tuple) for serialization.
    fn to_tuple(&self) -> (f64, Vec<(String, f64)>) {
        let items: Vec<(String, f64)> = self.units.iter().map(|(k, &v)| (k.clone(), v)).collect();
        (self.magnitude, items)
    }

    /// Construct Quantity from tuple (magnitude, ((unit, exp), ...))
    #[staticmethod]
    fn from_tuple(tup: (f64, Vec<(String, f64)>)) -> PyResult<Self> {
        let (mag, items) = tup;
        let mut uc = UnitsContainer::new();
        for (name, exp) in items {
            uc = uc.add(&name, exp);
        }
        let reg = RustRegistry::new().to_py()?;
        Ok(Self {
            magnitude: mag,
            units: uc,
            registry: Arc::new(Mutex::new(reg)),
            cached_dim: OnceCell::new(),
            cached_registry_obj: OnceCell::new(),
        })
    }

    /// Return unit_name -> exponent items.
    fn unit_items(&self) -> Vec<(String, f64)> {
        self.units.iter().map(|(k, &v)| (k.clone(), v)).collect()
    }

    /// Compare two quantities with a given operator string.
    fn compare(&self, other: &Bound<'_, PyAny>, op: &str) -> PyResult<bool> {
        match op {
            "eq" | "==" => self.__eq__(other),
            "ne" | "!=" => self.__ne__(other),
            "lt" | "<" => self.__lt__(other),
            "le" | "<=" => self.__le__(other),
            "gt" | ">" => self.__gt__(other),
            "ge" | ">=" => self.__ge__(other),
            _ => Err(PyValueError::new_err(format!("Unknown operator: {}", op))),
        }
    }

    /// Convert to datetime.timedelta (for time quantities only).
    fn to_timedelta(&self) -> PyResult<PyObject> {
        let mut reg = self.registry.lock().unwrap();
        let seconds_uc = reg.parse_unit_expr("second").to_py()?;
        let seconds = reg
            .convert(self.magnitude, &self.units, &seconds_uc)
            .to_py()?;
        Python::with_gil(|py| {
            let datetime = py.import("datetime")?;
            let td = datetime.getattr("timedelta")?.call1((0, seconds))?;
            Ok(td.unbind())
        })
    }

    /// Create a Measurement from this Quantity with given error.
    #[pyo3(signature = (error, relative=false))]
    fn plus_minus(&self, error: f64, relative: bool) -> PyResult<PyObject> {
        let actual_error = if relative {
            self.magnitude.abs() * error
        } else {
            error
        };
        Python::with_gil(|py| {
            let pintrs = py.import("pintrs")?;
            let measurement_cls = pintrs.getattr("Measurement")?;
            let self_obj = self.clone().into_pyobject(py)?;
            let m = measurement_cls.call1((self_obj, actual_error))?;
            Ok(m.unbind())
        })
    }

    #[getter]
    fn dimensionality(&self) -> PyResult<String> {
        if let Some(cached) = self.cached_dim.get() {
            return Ok(cached.clone());
        }
        let mut reg = self.registry.lock().unwrap();
        let dim = reg.get_dimensionality(&self.units).to_py()?;
        let s = dim.to_string();
        let _ = self.cached_dim.set(s.clone());
        Ok(s)
    }

    #[getter]
    fn dimensionless(&self) -> PyResult<bool> {
        let dim_str = self.dimensionality()?;
        Ok(dim_str == "dimensionless")
    }

    #[getter]
    fn unitless(&self) -> PyResult<bool> {
        self.dimensionless()
    }

    fn check(&self, dimension: &str) -> PyResult<bool> {
        let mut reg = self.registry.lock().unwrap();
        let self_dim = reg.get_dimensionality(&self.units).to_py()?;
        Ok(self_dim.to_string() == dimension)
    }

    fn is_compatible_with(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let mut reg = self.registry.lock().unwrap();
        let self_dim = reg.get_dimensionality(&self.units).to_py()?;
        if let Ok(other_q) = other.extract::<PyQuantity>() {
            let other_dim = reg.get_dimensionality(&other_q.units).to_py()?;
            Ok(self_dim == other_dim)
        } else if let Ok(other_u) = other.extract::<PyUnit>() {
            let other_dim = reg.get_dimensionality(&other_u.units).to_py()?;
            Ok(self_dim == other_dim)
        } else if let Ok(s) = other.extract::<String>() {
            let other_uc = reg.parse_unit_expr(&s).to_py()?;
            let other_dim = reg.get_dimensionality(&other_uc).to_py()?;
            Ok(self_dim == other_dim)
        } else {
            Ok(self_dim.is_empty())
        }
    }

    fn compatible_units(&self) -> PyResult<Vec<String>> {
        let mut reg = self.registry.lock().unwrap();
        let dim = reg.get_dimensionality(&self.units).to_py()?;
        Ok(reg.get_units_by_dimensionality(&dim))
    }

    /// Internal: get unit string using symbols (compact form).
    /// The Python formatting layer uses this to obtain the abbreviated unit names.
    fn _units_str(&self) -> String {
        let reg = self.registry.lock().unwrap();
        reg.format_units_compact(&self.units)
    }

    /// Internal: get the raw units container as dict
    fn _units_dict(&self) -> Vec<(String, f64)> {
        self.units.iter().map(|(k, &v)| (k.clone(), v)).collect()
    }

    fn __repr__(&self) -> String {
        let reg = self.registry.lock().unwrap();
        let unit_str = reg.format_units(&self.units);
        if (self.magnitude - self.magnitude.round()).abs() < f64::EPSILON
            && self.magnitude.abs() < 1e15
        {
            format!("<Quantity({}, '{}')>", self.magnitude, unit_str)
        } else {
            format!("<Quantity({:.9}, '{}')>", self.magnitude, unit_str)
        }
    }

    fn __str__(&self) -> String {
        let reg = self.registry.lock().unwrap();
        let unit_str = reg.format_units(&self.units);
        format!("{} {}", self.magnitude, unit_str)
    }

    fn __format__(&self, spec: &str) -> PyResult<String> {
        let reg = self.registry.lock().unwrap();
        let compact = spec.contains('~');
        let unit_str = if compact {
            reg.format_units_compact(&self.units)
        } else {
            reg.format_units(&self.units)
        };

        if spec.is_empty() {
            return Ok(format!("{} {}", self.magnitude, unit_str));
        }

        let spec_clean: String = spec.chars().filter(|c| *c != '~').collect();

        // Extract unit format type (last char if it's a letter and not part of float format)
        let (mag_spec, _unit_fmt) = if spec_clean.ends_with('P')
            || spec_clean.ends_with('L')
            || spec_clean.ends_with('H')
            || spec_clean.ends_with('C')
            || spec_clean.ends_with('D')
        {
            let (m, u) = spec_clean.split_at(spec_clean.len() - 1);
            (m, u)
        } else {
            (spec_clean.as_str(), "D")
        };

        let mag_str = if mag_spec.is_empty() {
            format!("{}", self.magnitude)
        } else {
            // Use Python's string formatting for magnitude
            format!("{}", self.magnitude) // Simplified; full impl in Python wrapper
        };

        Ok(format!("{} {}", mag_str, unit_str))
    }

    fn __float__(&self) -> PyResult<f64> {
        let mut reg = self.registry.lock().unwrap();
        let dim = reg.get_dimensionality(&self.units).to_py()?;
        if !dim.is_empty() {
            return Err(DimensionalityError::new_err(format!(
                "Cannot convert Quantity with dimensionality {} to float",
                dim
            )));
        }
        let factor = reg
            .get_conversion_factor(&self.units, &UnitsContainer::new())
            .to_py()?;
        Ok(self.magnitude * factor)
    }

    fn __int__(&self) -> PyResult<i64> {
        Ok(self.__float__()? as i64)
    }

    fn __bool__(&self) -> bool {
        self.magnitude != 0.0
    }

    fn __neg__(&self) -> PyQuantity {
        self.rebuild(-self.magnitude, self.units.clone())
    }

    fn __abs__(&self) -> PyQuantity {
        self.rebuild(self.magnitude.abs(), self.units.clone())
    }

    #[pyo3(signature = (ndigits=None))]
    fn __round__(&self, ndigits: Option<i32>) -> PyQuantity {
        let n = ndigits.unwrap_or(0);
        let factor = 10f64.powi(n);
        self.rebuild(
            (self.magnitude * factor).round() / factor,
            self.units.clone(),
        )
    }

    fn __add__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyQuantity> {
        if let Ok(other_q) = other.extract::<PyQuantity>() {
            // Offset/log (non-multiplicative) units follow special add rules.
            {
                let mut reg = self.registry.lock().unwrap();
                if reg.is_non_multiplicative(&self.units)
                    || reg.is_non_multiplicative(&other_q.units)
                {
                    let (mag, units) = reg
                        .offset_log_add_sub(
                            &self.units,
                            self.magnitude,
                            &other_q.units,
                            other_q.magnitude,
                            false,
                        )
                        .to_py()?;
                    return Ok(self.rebuild(mag, units));
                }
            }
            if self.units == other_q.units {
                return Ok(self.rebuild(self.magnitude + other_q.magnitude, self.units.clone()));
            }
            let mut reg = self.registry.lock().unwrap();
            let factor = reg
                .get_conversion_factor(&other_q.units, &self.units)
                .to_py()?;
            Ok(self.rebuild(
                self.magnitude + other_q.magnitude * factor,
                self.units.clone(),
            ))
        } else if let Ok(val) = other.extract::<f64>() {
            if self.units.is_empty() || val == 0.0 {
                return Ok(self.rebuild(self.magnitude + val, self.units.clone()));
            }
            // Dimensionless but with non-empty (scaled) units, e.g. `cm*s/m/ms` or
            // `radian`. Convert the magnitude to plain dimensionless units before
            // adding the bare number, matching pint (which yields `dimensionless`,
            // not the intermediate scale or a self-rooting unit like `radian`).
            let value = self.dimensionless_value()?;
            Ok(self.rebuild(value + val, UnitsContainer::new()))
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for +"))
        }
    }

    fn __radd__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyQuantity> {
        self.__add__(other)
    }

    fn __sub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyQuantity> {
        if let Ok(other_q) = other.extract::<PyQuantity>() {
            // Offset/log (non-multiplicative) units follow special subtract rules
            // (absolute - absolute yields a delta).
            {
                let mut reg = self.registry.lock().unwrap();
                if reg.is_non_multiplicative(&self.units)
                    || reg.is_non_multiplicative(&other_q.units)
                {
                    let (mag, units) = reg
                        .offset_log_add_sub(
                            &self.units,
                            self.magnitude,
                            &other_q.units,
                            other_q.magnitude,
                            true,
                        )
                        .to_py()?;
                    return Ok(self.rebuild(mag, units));
                }
            }
            if self.units == other_q.units {
                return Ok(self.rebuild(self.magnitude - other_q.magnitude, self.units.clone()));
            }
            let mut reg = self.registry.lock().unwrap();
            let factor = reg
                .get_conversion_factor(&other_q.units, &self.units)
                .to_py()?;
            Ok(self.rebuild(
                self.magnitude - other_q.magnitude * factor,
                self.units.clone(),
            ))
        } else if let Ok(val) = other.extract::<f64>() {
            if self.units.is_empty() || val == 0.0 {
                return Ok(self.rebuild(self.magnitude - val, self.units.clone()));
            }
            // Dimensionless but with non-empty (scaled) units: convert to plain
            // dimensionless units before subtracting the bare number (pint parity).
            let value = self.dimensionless_value()?;
            Ok(self.rebuild(value - val, UnitsContainer::new()))
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for -"))
        }
    }

    fn __rsub__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyQuantity> {
        let result = self.__sub__(other)?;
        Ok(PyQuantity {
            magnitude: -result.magnitude,
            units: result.units,
            registry: result.registry,
            cached_dim: OnceCell::new(),
            cached_registry_obj: OnceCell::new(),
        })
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyQuantity> {
        // Multiplying a non-multiplicative (offset/log) unit is ambiguous; pint
        // raises OffsetUnitCalculusError. Checked before the numeric fast path so
        // `degC * 2` and `2 * degC` (via __rmul__) both raise.
        self.ensure_multiplicative(&self.units)?;
        // Check if it's a native Python float/int first (most common case).
        // Use is_instance_of to avoid triggering __float__ on Quantity objects.
        if other.is_instance_of::<pyo3::types::PyFloat>()
            || other.is_instance_of::<pyo3::types::PyInt>()
        {
            let val: f64 = other.extract()?;
            return Ok(self.rebuild(self.magnitude * val, self.units.clone()));
        }
        if let Ok(other_q) = other.extract::<PyQuantity>() {
            other_q.ensure_multiplicative(&other_q.units)?;
            Ok(self.rebuild(
                self.magnitude * other_q.magnitude,
                &self.units * &other_q.units,
            ))
        } else if let Ok(other_u) = other.extract::<PyUnit>() {
            self.ensure_multiplicative(&other_u.units)?;
            Ok(self.rebuild(self.magnitude, &self.units * &other_u.units))
        } else if let Ok(val) = other.extract::<f64>() {
            // Fallback for other numeric types (numpy scalars, etc.)
            Ok(self.rebuild(self.magnitude * val, self.units.clone()))
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for *"))
        }
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyQuantity> {
        self.__mul__(other)
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyQuantity> {
        // Dividing a non-multiplicative (offset/log) unit is ambiguous; pint
        // raises OffsetUnitCalculusError (e.g. `degC / 2`, `2 / degC`).
        self.ensure_multiplicative(&self.units)?;
        if other.is_instance_of::<pyo3::types::PyFloat>()
            || other.is_instance_of::<pyo3::types::PyInt>()
        {
            let val: f64 = other.extract()?;
            if val == 0.0 {
                return Err(PyZeroDivisionError::new_err("Division by zero"));
            }
            return Ok(self.rebuild(self.magnitude / val, self.units.clone()));
        }
        if let Ok(other_q) = other.extract::<PyQuantity>() {
            other_q.ensure_multiplicative(&other_q.units)?;
            if other_q.magnitude == 0.0 {
                return Err(PyZeroDivisionError::new_err("Division by zero"));
            }
            Ok(self.rebuild(
                self.magnitude / other_q.magnitude,
                &self.units / &other_q.units,
            ))
        } else if let Ok(other_u) = other.extract::<PyUnit>() {
            self.ensure_multiplicative(&other_u.units)?;
            Ok(self.rebuild(self.magnitude, &self.units / &other_u.units))
        } else if let Ok(val) = other.extract::<f64>() {
            if val == 0.0 {
                return Err(PyZeroDivisionError::new_err("Division by zero"));
            }
            Ok(self.rebuild(self.magnitude / val, self.units.clone()))
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for /"))
        }
    }

    fn __rtruediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyQuantity> {
        self.ensure_multiplicative(&self.units)?;
        if self.magnitude == 0.0 {
            return Err(PyZeroDivisionError::new_err("Division by zero"));
        }
        if let Ok(val) = other.extract::<f64>() {
            Ok(self.rebuild(val / self.magnitude, self.units.inv()))
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for /"))
        }
    }

    fn __floordiv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyQuantity> {
        if let Ok(other_q) = other.extract::<PyQuantity>() {
            if other_q.magnitude == 0.0 {
                return Err(PyZeroDivisionError::new_err("Division by zero"));
            }
            // Floor division of two quantities requires compatible dimensions
            // (the conversion fails otherwise) and yields a plain dimensionless
            // number computed from the converted values, matching pint.
            let mut reg = self.registry.lock().unwrap();
            let factor = reg
                .get_conversion_factor(&other_q.units, &self.units)
                .to_py()?;
            Ok(self.rebuild(
                floored_div(self.magnitude, other_q.magnitude * factor),
                UnitsContainer::new(),
            ))
        } else if let Ok(val) = other.extract::<f64>() {
            // Floor-dividing by a bare number requires a dimensionless quantity
            // and yields a plain number; operate in base units (pint parity).
            // The dimensionality check precedes the zero check, like pint.
            let value = self.dimensionless_value()?;
            if val == 0.0 {
                return Err(PyZeroDivisionError::new_err("Division by zero"));
            }
            Ok(self.rebuild(floored_div(value, val), UnitsContainer::new()))
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for //"))
        }
    }

    fn __rfloordiv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyQuantity> {
        if let Ok(val) = other.extract::<f64>() {
            // number // quantity: requires a dimensionless quantity; operate in
            // base units and return a plain number (pint parity).
            let base = self.dimensionless_value()?;
            if base == 0.0 {
                return Err(PyZeroDivisionError::new_err("Division by zero"));
            }
            Ok(self.rebuild(floored_div(val, base), UnitsContainer::new()))
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for //"))
        }
    }

    fn __mod__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyQuantity> {
        if let Ok(other_q) = other.extract::<PyQuantity>() {
            if other_q.magnitude == 0.0 {
                return Err(PyZeroDivisionError::new_err("Modulo by zero"));
            }
            if self.units == other_q.units {
                return Ok(self.rebuild(
                    floored_mod(self.magnitude, other_q.magnitude),
                    self.units.clone(),
                ));
            }
            let mut reg = self.registry.lock().unwrap();
            let factor = reg
                .get_conversion_factor(&other_q.units, &self.units)
                .to_py()?;
            Ok(self.rebuild(
                floored_mod(self.magnitude, other_q.magnitude * factor),
                self.units.clone(),
            ))
        } else if let Ok(val) = other.extract::<f64>() {
            // Modulo by a bare number requires a dimensionless quantity. pint keeps
            // the dividend's units but operates on the base-unit value; compute the
            // remainder in base units, then express it back in `self.units`. The
            // dimensionality check precedes the zero check, like pint.
            let factor = self.require_dimensionless_factor()?;
            if val == 0.0 {
                return Err(PyZeroDivisionError::new_err("Modulo by zero"));
            }
            let base_remainder = floored_mod(self.magnitude * factor, val);
            Ok(self.rebuild(base_remainder / factor, self.units.clone()))
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for %"))
        }
    }

    fn __rmod__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyQuantity> {
        if let Ok(val) = other.extract::<f64>() {
            // number % quantity: requires a dimensionless quantity; operate in base
            // units and return a plain number (pint parity).
            let base = self.dimensionless_value()?;
            if base == 0.0 {
                return Err(PyZeroDivisionError::new_err("Modulo by zero"));
            }
            Ok(self.rebuild(floored_mod(val, base), UnitsContainer::new()))
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for %"))
        }
    }

    fn __divmod__(&self, other: &Bound<'_, PyAny>) -> PyResult<(PyQuantity, PyQuantity)> {
        // `divmod(q, x) == (q // x, q % x)`, reusing the consistent floor/mod logic.
        Ok((self.__floordiv__(other)?, self.__mod__(other)?))
    }

    fn __rdivmod__(&self, other: &Bound<'_, PyAny>) -> PyResult<(PyQuantity, PyQuantity)> {
        Ok((self.__rfloordiv__(other)?, self.__rmod__(other)?))
    }

    fn __pow__(&self, exp: f64, _modulo: Option<u32>) -> PyResult<PyQuantity> {
        // Raising a non-multiplicative (offset/log) unit to a power other than
        // 0 or 1 is ambiguous; pint raises OffsetUnitCalculusError. `**0` yields
        // a dimensionless 1 and `**1` is the identity, both of which are allowed.
        if exp != 0.0 && exp != 1.0 {
            self.ensure_multiplicative(&self.units)?;
        }
        Ok(self.rebuild(self.magnitude.powf(exp), self.units.pow(exp)))
    }

    fn __rpow__(&self, base: f64, _modulo: Option<u32>) -> PyResult<f64> {
        // `number ** quantity` requires a dimensionless exponent (taken at its
        // base value) and yields a plain number, matching pint.
        let value = self.dimensionless_value()?;
        Ok(base.powf(value))
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        if let Ok(other_q) = other.extract::<PyQuantity>() {
            if self.units == other_q.units {
                return Ok((self.magnitude - other_q.magnitude).abs()
                    < f64::EPSILON * self.magnitude.abs().max(1.0));
            }
            let mut reg = self.registry.lock().unwrap();
            // Use the offset/log-aware conversion so e.g. `0 dBW == 1 W`.
            match reg.convert(other_q.magnitude, &other_q.units, &self.units) {
                Ok(converted) => Ok((self.magnitude - converted).abs()
                    < f64::EPSILON * self.magnitude.abs().max(1.0)),
                Err(_) => Ok(false),
            }
        } else if let Ok(val) = other.extract::<f64>() {
            if self.units.is_empty() {
                return Ok((self.magnitude - val).abs() < f64::EPSILON * val.abs().max(1.0));
            }
            let mut reg = self.registry.lock().unwrap();
            let dim = reg.get_dimensionality(&self.units).to_py()?;
            if dim.is_empty() {
                // Offset/log-aware so e.g. `1 dB == 10**0.1`, matching pint.
                let converted = reg
                    .convert(self.magnitude, &self.units, &UnitsContainer::new())
                    .to_py()?;
                Ok((converted - val).abs() < f64::EPSILON * val.abs().max(1.0))
            } else {
                Ok(false)
            }
        } else {
            Ok(false)
        }
    }

    fn __ne__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.__eq__(other).map(|v| !v)
    }

    fn __lt__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.compare_ordered(other, |a, b| a < b)
    }

    fn __le__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.compare_ordered(other, |a, b| a <= b)
    }

    fn __gt__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.compare_ordered(other, |a, b| a > b)
    }

    fn __ge__(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.compare_ordered(other, |a, b| a >= b)
    }

    fn __hash__(&self) -> PyResult<isize> {
        let base = self.to_base_units()?;
        let mut reg = self.registry.lock().unwrap();
        let dim = reg.get_dimensionality(&base.units).to_py()?;
        if dim.is_empty() {
            Ok(base.magnitude.to_bits() as isize)
        } else {
            let mut h: u64 = base.magnitude.to_bits();
            for (k, &v) in base.units.iter() {
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                k.hash(&mut hasher);
                v.to_bits().hash(&mut hasher);
                h ^= hasher.finish();
            }
            Ok(h as isize)
        }
    }

    fn __reduce__(&self) -> PyResult<(PyObject, (f64, String))> {
        Python::with_gil(|py| {
            let cls = py.get_type::<PyQuantity>();
            let reg = self.registry.lock().unwrap();
            let units_str = reg.format_units(&self.units);
            Ok((cls.unbind().into_any(), (self.magnitude, units_str)))
        })
    }

    fn __copy__(&self) -> PyQuantity {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> PyQuantity {
        self.clone()
    }
}

// --- Unit ---

#[pyclass(name = "Unit", module = "pintrs._core", subclass)]
pub(crate) struct PyUnit {
    pub(crate) units: UnitsContainer,
    pub(crate) registry: SharedRegistry,
    pub(crate) cached_registry_obj: OnceCell<PyObject>,
}

impl Clone for PyUnit {
    fn clone(&self) -> Self {
        Self {
            units: self.units.clone(),
            registry: Arc::clone(&self.registry),
            cached_registry_obj: OnceCell::new(),
        }
    }
}

impl PyUnit {
    /// Raise `OffsetUnitCalculusError` if these units are non-multiplicative.
    /// Used when a `Unit` is scaled by a number (`degC * 2`), which produces a
    /// `Quantity` and is forbidden by pint for offset/log units.
    fn ensure_multiplicative(&self) -> PyResult<()> {
        let reg = self.registry.lock().unwrap();
        ensure_units_multiplicative(&reg, &self.units)
    }
}

#[pymethods]
impl PyUnit {
    #[new]
    fn new(units: &str) -> PyResult<Self> {
        let mut reg = RustRegistry::new().to_py()?;
        let uc = reg.parse_unit_expr(units).to_py()?;
        let shared = Arc::new(Mutex::new(reg));
        Ok(Self {
            units: uc,
            registry: shared,
            cached_registry_obj: OnceCell::new(),
        })
    }

    #[getter]
    fn _registry(&self, py: Python<'_>) -> PyResult<PyObject> {
        registry_pyobject(py, &self.registry, &self.cached_registry_obj)
    }

    #[getter]
    fn dimensionality(&self) -> PyResult<String> {
        let mut reg = self.registry.lock().unwrap();
        let dim = reg.get_dimensionality(&self.units).to_py()?;
        Ok(dim.to_string())
    }

    #[getter]
    fn dimensionless(&self) -> PyResult<bool> {
        let mut reg = self.registry.lock().unwrap();
        let dim = reg.get_dimensionality(&self.units).to_py()?;
        Ok(dim.is_empty())
    }

    fn is_compatible_with(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let mut reg = self.registry.lock().unwrap();
        let self_dim = reg.get_dimensionality(&self.units).to_py()?;
        if let Ok(other_u) = other.extract::<PyUnit>() {
            let other_dim = reg.get_dimensionality(&other_u.units).to_py()?;
            Ok(self_dim == other_dim)
        } else if let Ok(s) = other.extract::<String>() {
            let other_uc = reg.parse_unit_expr(&s).to_py()?;
            let other_dim = reg.get_dimensionality(&other_uc).to_py()?;
            Ok(self_dim == other_dim)
        } else {
            Ok(false)
        }
    }

    fn compatible_units(&self) -> PyResult<Vec<String>> {
        let mut reg = self.registry.lock().unwrap();
        let dim = reg.get_dimensionality(&self.units).to_py()?;
        Ok(reg.get_units_by_dimensionality(&dim))
    }

    fn _units_str(&self) -> String {
        let reg = self.registry.lock().unwrap();
        reg.format_units_compact(&self.units)
    }

    fn _units_dict(&self) -> Vec<(String, f64)> {
        self.units.iter().map(|(k, &v)| (k.clone(), v)).collect()
    }

    fn __repr__(&self) -> String {
        let reg = self.registry.lock().unwrap();
        format!("<Unit('{}')>", reg.format_units(&self.units))
    }

    fn __str__(&self) -> String {
        let reg = self.registry.lock().unwrap();
        reg.format_units(&self.units)
    }

    fn __format__(&self, spec: &str) -> PyResult<String> {
        let reg = self.registry.lock().unwrap();
        if spec.contains('~') {
            Ok(reg.format_units_compact(&self.units))
        } else {
            Ok(reg.format_units(&self.units))
        }
    }

    fn __mul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let py = other.py();
        let propagate_reg = || -> OnceCell<PyObject> {
            let c = OnceCell::new();
            if let Some(r) = self.cached_registry_obj.get() {
                let _ = c.set(r.clone_ref(py));
            }
            c
        };
        if let Ok(other_u) = other.extract::<PyUnit>() {
            Ok(PyUnit {
                units: &self.units * &other_u.units,
                registry: Arc::clone(&self.registry),
                cached_registry_obj: propagate_reg(),
            }
            .into_pyobject(py)?
            .into_any()
            .unbind())
        } else if let Ok(val) = other.extract::<f64>() {
            self.ensure_multiplicative()?;
            Ok(PyQuantity {
                magnitude: val,
                units: self.units.clone(),
                registry: Arc::clone(&self.registry),
                cached_dim: OnceCell::new(),
                cached_registry_obj: propagate_reg(),
            }
            .into_pyobject(py)?
            .into_any()
            .unbind())
        } else if let Ok(val) = other.extract::<i64>() {
            self.ensure_multiplicative()?;
            Ok(PyQuantity {
                magnitude: val as f64,
                units: self.units.clone(),
                registry: Arc::clone(&self.registry),
                cached_dim: OnceCell::new(),
                cached_registry_obj: propagate_reg(),
            }
            .into_pyobject(py)?
            .into_any()
            .unbind())
        } else {
            Ok(py.NotImplemented())
        }
    }

    fn __rmul__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        self.__mul__(other)
    }

    fn __truediv__(&self, other: &Bound<'_, PyAny>) -> PyResult<PyUnit> {
        if let Ok(other_u) = other.extract::<PyUnit>() {
            Ok(PyUnit {
                units: &self.units / &other_u.units,
                registry: Arc::clone(&self.registry),
                cached_registry_obj: OnceCell::new(),
            })
        } else {
            Err(PyTypeError::new_err("Unsupported operand type for /"))
        }
    }

    fn __pow__(&self, exp: f64, _modulo: Option<u32>) -> PyUnit {
        PyUnit {
            units: self.units.pow(exp),
            registry: Arc::clone(&self.registry),
            cached_registry_obj: OnceCell::new(),
        }
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        if let Ok(other_u) = other.extract::<PyUnit>() {
            self.units == other_u.units
        } else {
            false
        }
    }

    fn __hash__(&self) -> isize {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.units.hash(&mut hasher);
        hasher.finish() as isize
    }

    fn __reduce__(&self) -> PyResult<(PyObject, (String,))> {
        Python::with_gil(|py| {
            let cls = py.get_type::<PyUnit>();
            let reg = self.registry.lock().unwrap();
            let units_str = reg.format_units(&self.units);
            Ok((cls.unbind().into_any(), (units_str,)))
        })
    }

    /// Convert a Quantity to this Unit.
    fn from_(&self, value: &Bound<'_, PyAny>) -> PyResult<PyQuantity> {
        let q: PyQuantity = value.extract()?;
        let mut reg = self.registry.lock().unwrap();
        let new_mag = reg.convert(q.magnitude, &q.units, &self.units).to_py()?;
        Ok(PyQuantity::from_parts(
            new_mag,
            self.units.clone(),
            Arc::clone(&self.registry),
        ))
    }

    /// Get magnitude of a Quantity in this Unit.
    fn m_from(&self, value: &Bound<'_, PyAny>) -> PyResult<f64> {
        let q: PyQuantity = value.extract()?;
        let mut reg = self.registry.lock().unwrap();
        reg.convert(q.magnitude, &q.units, &self.units).to_py()
    }

    /// Return set of system names this unit belongs to (stub: returns empty set).
    #[getter]
    fn systems(&self) -> Vec<String> {
        Vec::new()
    }

    /// Compare to another Unit
    fn compare(&self, other: &Bound<'_, PyAny>, op: &str) -> PyResult<bool> {
        match op {
            "eq" | "==" => Ok(self.__eq__(other)),
            "ne" | "!=" => Ok(!self.__eq__(other)),
            _ => Err(PyValueError::new_err(format!("Unknown operator: {}", op))),
        }
    }

    fn __copy__(&self) -> PyUnit {
        self.clone()
    }
    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> PyUnit {
        self.clone()
    }
}

// --- helpers ---

/// Extract a unit string from a Python object that can be a str, Unit, or Quantity.
pub(crate) fn extract_units_string(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    if let Ok(s) = obj.extract::<String>() {
        return Ok(s);
    }
    if let Ok(u) = obj.extract::<PyUnit>() {
        let reg = u.registry.lock().unwrap();
        return Ok(reg.format_units(&u.units));
    }
    if let Ok(q) = obj.extract::<PyQuantity>() {
        let reg = q.registry.lock().unwrap();
        return Ok(reg.format_units(&q.units));
    }
    Err(PyTypeError::new_err(
        "units must be a string, Unit, or Quantity",
    ))
}

fn parse_quantity_string(s: &str) -> (String, &str) {
    let s = s.trim();
    let mut split_pos = 0;
    let chars: Vec<char> = s.chars().collect();

    while split_pos < chars.len() && chars[split_pos].is_whitespace() {
        split_pos += 1;
    }
    if split_pos < chars.len() && (chars[split_pos] == '+' || chars[split_pos] == '-') {
        split_pos += 1;
    }

    let mut seen_dot = false;
    let mut seen_e = false;
    while split_pos < chars.len() {
        let c = chars[split_pos];
        if c.is_ascii_digit() {
            split_pos += 1;
        } else if c == '.' && !seen_dot && !seen_e {
            seen_dot = true;
            split_pos += 1;
        } else if (c == 'e' || c == 'E') && !seen_e {
            seen_e = true;
            split_pos += 1;
            if split_pos < chars.len() && (chars[split_pos] == '+' || chars[split_pos] == '-') {
                split_pos += 1;
            }
        } else {
            break;
        }
    }

    if split_pos == 0 {
        return ("1".to_string(), s);
    }

    let magnitude: String = chars[..split_pos].iter().collect();
    let rest = s[magnitude.len()..].trim();
    (magnitude, rest)
}

// --- Module ---

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyUnitRegistry>()?;
    m.add_class::<PyQuantity>()?;
    m.add_class::<PyUnit>()?;
    m.add_class::<array_quantity::PyArrayQuantity>()?;

    // Exception types
    m.add("PintError", m.py().get_type::<PintError_Py>())?;
    m.add(
        "DimensionalityError",
        m.py().get_type::<DimensionalityError>(),
    )?;
    m.add(
        "UndefinedUnitError",
        m.py().get_type::<UndefinedUnitError>(),
    )?;
    m.add(
        "OffsetUnitCalculusError",
        m.py().get_type::<OffsetUnitCalculusError>(),
    )?;
    m.add(
        "DefinitionSyntaxError",
        m.py().get_type::<DefinitionSyntaxError>(),
    )?;
    m.add("RedefinitionError", m.py().get_type::<RedefinitionError>())?;

    Ok(())
}
