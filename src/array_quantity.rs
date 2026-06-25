//! Rust-backed ArrayQuantity for fast numpy array + unit operations.

use std::sync::{Arc, Mutex};

use numpy::{PyArray1, PyArrayMethods, PyUntypedArrayMethods};
use once_cell::sync::OnceCell;
use pyo3::prelude::*;

use crate::registry::UnitRegistry as RustRegistry;
use crate::units_container::UnitsContainer;
use crate::{
    ensure_units_multiplicative, registry_pyobject, PintResultExt, PyQuantity, PyUnit,
    PyUnitRegistry, SharedRegistry,
};

/// Helper: multiply a numpy array by a scalar, returning a new array.
fn np_mul_scalar<'py>(
    arr: &Bound<'py, PyArray1<f64>>,
    factor: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let result = arr.call_method1("__mul__", (factor,))?;
    result.downcast_into().map_err(Into::into)
}

/// Convert an array's magnitude from `src` to `dst` units. Ordinary
/// (multiplicative) units use a single factor; offset/logarithmic units apply
/// the offset/log transform element-wise, matching scalar `Quantity.to()`.
fn convert_array_magnitude<'py>(
    arr: &Bound<'py, PyArray1<f64>>,
    reg: &mut RustRegistry,
    src: &UnitsContainer,
    dst: &UnitsContainer,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    if reg.has_offset_units(src)
        || reg.has_log_units(src)
        || reg.has_offset_units(dst)
        || reg.has_log_units(dst)
    {
        let values = arr.to_vec()?;
        let mut out = Vec::with_capacity(values.len());
        for v in values {
            out.push(reg.convert(v, src, dst).to_py()?);
        }
        Ok(PyArray1::from_vec(arr.py(), out))
    } else {
        let factor = reg.get_conversion_factor(src, dst).to_py()?;
        np_mul_scalar(arr, factor)
    }
}

/// A Quantity whose magnitude is a numpy f64 array.
#[pyclass(name = "RustArrayQuantity", module = "pintrs._core", subclass)]
pub struct PyArrayQuantity {
    magnitude: Py<PyArray1<f64>>,
    units: UnitsContainer,
    registry: SharedRegistry,
    cached_registry_obj: OnceCell<PyObject>,
}

impl PyArrayQuantity {
    /// Raise `OffsetUnitCalculusError` if `units` is non-multiplicative. Pass
    /// `&self.units` to guard the receiver or an operand's units to guard the
    /// other side of `*`/`/`, matching the scalar `Quantity` behaviour.
    fn ensure_multiplicative(&self, units: &UnitsContainer) -> PyResult<()> {
        let reg = self.registry.lock().unwrap();
        ensure_units_multiplicative(&reg, units)
    }

    /// Build a new array quantity from a result magnitude and units, sharing
    /// this one's registry. Centralises the construction boilerplate.
    #[inline]
    fn rebuild(&self, magnitude: Bound<'_, PyArray1<f64>>, units: UnitsContainer) -> Self {
        Self {
            magnitude: magnitude.unbind(),
            units,
            registry: Arc::clone(&self.registry),
            cached_registry_obj: OnceCell::new(),
        }
    }

    /// Like [`rebuild`](Self::rebuild) but returns the boxed Python object, the
    /// shape every operator branch needs.
    #[inline]
    fn wrap(
        &self,
        py: Python<'_>,
        magnitude: Bound<'_, PyArray1<f64>>,
        units: UnitsContainer,
    ) -> PyResult<PyObject> {
        Ok(self
            .rebuild(magnitude, units)
            .into_pyobject(py)?
            .into_any()
            .unbind())
    }

    /// Multiplicative factor that brings a value in `units` into `self.units`.
    /// `1.0` (no conversion) when the units already match.
    fn conversion_factor_from(&self, units: &UnitsContainer) -> PyResult<f64> {
        if self.units == *units {
            return Ok(1.0);
        }
        let mut reg = self.registry.lock().unwrap();
        reg.get_conversion_factor(units, &self.units).to_py()
    }

    /// Shared body of `__add__`/`__sub__`; `op` is the numpy dunder to apply.
    /// The other operand (array quantity or scalar quantity) is converted into
    /// `self`'s units first, leaving the result in those units.
    fn add_sub(&self, py: Python<'_>, other: &Bound<'_, PyAny>, op: &str) -> PyResult<PyObject> {
        let a = self.magnitude.bind(py);
        if let Ok(other_aq) = other.extract::<PyRef<'_, Self>>() {
            let b = other_aq.magnitude.bind(py);
            let factor = self.conversion_factor_from(&other_aq.units)?;
            let result: Bound<'_, PyArray1<f64>> = if factor == 1.0 {
                a.call_method1(op, (b,))?.downcast_into()?
            } else {
                let b_scaled = np_mul_scalar(b, factor)?;
                a.call_method1(op, (&b_scaled,))?.downcast_into()?
            };
            return self.wrap(py, result, self.units.clone());
        }
        if let Ok(q) = other.extract::<PyQuantity>() {
            let factor = self.conversion_factor_from(&q.units)?;
            let result: Bound<'_, PyArray1<f64>> = a
                .call_method1(op, (q.magnitude * factor,))?
                .downcast_into()?;
            return self.wrap(py, result, self.units.clone());
        }
        Ok(py.NotImplemented())
    }
}

#[pymethods]
impl PyArrayQuantity {
    #[new]
    #[pyo3(signature = (magnitude, units, registry=None))]
    fn new(
        _py: Python<'_>,
        magnitude: &Bound<'_, PyAny>,
        units: &str,
        registry: Option<&Bound<'_, PyUnitRegistry>>,
    ) -> PyResult<Self> {
        let arr: &Bound<'_, PyArray1<f64>> = magnitude.downcast()?;
        let (shared, cached) = if let Some(reg_bound) = registry {
            let reg_ref = reg_bound.borrow();
            let c = OnceCell::new();
            let _ = c.set(reg_bound.as_any().clone().unbind());
            (Arc::clone(&reg_ref.inner), c)
        } else {
            let reg = RustRegistry::new().to_py()?;
            (Arc::new(Mutex::new(reg)), OnceCell::new())
        };

        let uc = {
            let mut reg = shared.lock().unwrap();
            reg.parse_unit_expr(units).to_py()?
        };

        Ok(Self {
            magnitude: arr.to_owned().unbind(),
            units: uc,
            registry: shared,
            cached_registry_obj: cached,
        })
    }

    #[getter]
    fn magnitude<'py>(&self, py: Python<'py>) -> &Bound<'py, PyArray1<f64>> {
        self.magnitude.bind(py)
    }

    #[getter]
    fn m<'py>(&self, py: Python<'py>) -> &Bound<'py, PyArray1<f64>> {
        self.magnitude.bind(py)
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

    #[getter]
    fn units_str(&self) -> String {
        let reg = self.registry.lock().unwrap();
        reg.format_units(&self.units)
    }

    #[getter]
    fn _units_str(&self) -> String {
        self.units_str()
    }

    #[getter]
    fn shape(&self, py: Python<'_>) -> Vec<usize> {
        self.magnitude.bind(py).shape().to_vec()
    }

    #[getter]
    fn ndim(&self, py: Python<'_>) -> usize {
        self.magnitude.bind(py).ndim()
    }

    #[getter]
    fn dtype(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(self.magnitude.bind(py).dtype().into_any().unbind())
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

    #[getter]
    fn _registry(&self, py: Python<'_>) -> PyResult<PyObject> {
        registry_pyobject(py, &self.registry, &self.cached_registry_obj)
    }

    #[getter]
    #[allow(non_snake_case)]
    fn _REGISTRY(&self, py: Python<'_>) -> PyResult<PyObject> {
        self._registry(py)
    }

    // --- Conversion ---

    fn to(&self, py: Python<'_>, units: &Bound<'_, PyAny>) -> PyResult<Self> {
        let dst_str = crate::extract_units_string(units)?;
        let mut reg = self.registry.lock().unwrap();
        let dst = reg.parse_unit_expr(&dst_str).to_py()?;
        let result = convert_array_magnitude(self.magnitude.bind(py), &mut reg, &self.units, &dst)?;
        drop(reg);
        Ok(self.rebuild(result, dst))
    }

    fn m_as(&self, py: Python<'_>, units: &Bound<'_, PyAny>) -> PyResult<Py<PyArray1<f64>>> {
        let dst_str = crate::extract_units_string(units)?;
        let mut reg = self.registry.lock().unwrap();
        let dst = reg.parse_unit_expr(&dst_str).to_py()?;
        let result = convert_array_magnitude(self.magnitude.bind(py), &mut reg, &self.units, &dst)?;
        drop(reg);
        Ok(result.unbind())
    }

    fn to_base_units(&self, py: Python<'_>) -> PyResult<Self> {
        let mut reg = self.registry.lock().unwrap();
        let (_factor, root_units) = reg.get_root_units(&self.units).to_py()?;
        let result =
            convert_array_magnitude(self.magnitude.bind(py), &mut reg, &self.units, &root_units)?;
        drop(reg);
        Ok(self.rebuild(result, root_units))
    }

    // --- Reductions ---

    fn sum(&self, py: Python<'_>) -> PyResult<PyQuantity> {
        let total: f64 = self.magnitude.bind(py).call_method0("sum")?.extract()?;
        Ok(PyQuantity::from_parts(
            total,
            self.units.clone(),
            Arc::clone(&self.registry),
        ))
    }

    fn mean(&self, py: Python<'_>) -> PyResult<PyQuantity> {
        let mean: f64 = self.magnitude.bind(py).call_method0("mean")?.extract()?;
        Ok(PyQuantity::from_parts(
            mean,
            self.units.clone(),
            Arc::clone(&self.registry),
        ))
    }

    // --- Dunder methods ---

    fn __len__(&self, py: Python<'_>) -> usize {
        self.magnitude.bind(py).len()
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let arr = self.magnitude.bind(py);
        let reg = self.registry.lock().unwrap();
        format!(
            "<ArrayQuantity({:?}, '{}')>",
            arr,
            reg.format_units(&self.units)
        )
    }

    fn __str__(&self, py: Python<'_>) -> String {
        let arr = self.magnitude.bind(py);
        let reg = self.registry.lock().unwrap();
        format!("{:?} {}", arr, reg.format_units(&self.units))
    }

    fn __mul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        self.ensure_multiplicative(&self.units)?;
        let arr = self.magnitude.bind(py);
        if let Ok(val) = other.extract::<f64>() {
            let result: Bound<'_, PyArray1<f64>> =
                arr.call_method1("__mul__", (val,))?.downcast_into()?;
            return self.wrap(py, result, self.units.clone());
        }
        if let Ok(other_aq) = other.extract::<PyRef<'_, Self>>() {
            self.ensure_multiplicative(&other_aq.units)?;
            let b = other_aq.magnitude.bind(py);
            let result: Bound<'_, PyArray1<f64>> =
                arr.call_method1("__mul__", (b,))?.downcast_into()?;
            return self.wrap(py, result, &self.units * &other_aq.units);
        }
        if let Ok(q) = other.extract::<PyQuantity>() {
            self.ensure_multiplicative(&q.units)?;
            let result: Bound<'_, PyArray1<f64>> = arr
                .call_method1("__mul__", (q.magnitude,))?
                .downcast_into()?;
            return self.wrap(py, result, &self.units * &q.units);
        }
        Ok(py.NotImplemented())
    }

    fn __rmul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        self.__mul__(py, other)
    }

    fn __add__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        self.add_sub(py, other, "__add__")
    }

    fn __radd__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        self.__add__(py, other)
    }

    fn __sub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        self.add_sub(py, other, "__sub__")
    }

    fn __rsub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        // other - self
        let a = self.magnitude.bind(py);
        if let Ok(q) = other.extract::<PyQuantity>() {
            let factor = self.conversion_factor_from(&q.units)?;
            let neg_a: Bound<'_, PyArray1<f64>> = a.call_method0("__neg__")?.downcast_into()?;
            let result: Bound<'_, PyArray1<f64>> = neg_a
                .call_method1("__add__", (q.magnitude * factor,))?
                .downcast_into()?;
            return self.wrap(py, result, self.units.clone());
        }
        Ok(py.NotImplemented())
    }

    fn __truediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        self.ensure_multiplicative(&self.units)?;
        let arr = self.magnitude.bind(py);
        if let Ok(val) = other.extract::<f64>() {
            if val == 0.0 {
                return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                    "division by zero",
                ));
            }
            let result: Bound<'_, PyArray1<f64>> =
                arr.call_method1("__truediv__", (val,))?.downcast_into()?;
            return self.wrap(py, result, self.units.clone());
        }
        if let Ok(other_aq) = other.extract::<PyRef<'_, Self>>() {
            self.ensure_multiplicative(&other_aq.units)?;
            let b = other_aq.magnitude.bind(py);
            let result: Bound<'_, PyArray1<f64>> =
                arr.call_method1("__truediv__", (b,))?.downcast_into()?;
            return self.wrap(py, result, &self.units / &other_aq.units);
        }
        if let Ok(q) = other.extract::<PyQuantity>() {
            self.ensure_multiplicative(&q.units)?;
            if q.magnitude == 0.0 {
                return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                    "division by zero",
                ));
            }
            let result: Bound<'_, PyArray1<f64>> = arr
                .call_method1("__truediv__", (q.magnitude,))?
                .downcast_into()?;
            return self.wrap(py, result, &self.units / &q.units);
        }
        if let Ok(u) = other.extract::<PyUnit>() {
            self.ensure_multiplicative(&u.units)?;
            // array / Unit -> divide magnitudes by 1 (unit only), adjust units
            return Ok(Self {
                magnitude: self.magnitude.clone_ref(py),
                units: &self.units / &u.units,
                registry: Arc::clone(&self.registry),
                cached_registry_obj: OnceCell::new(),
            }
            .into_pyobject(py)?
            .into_any()
            .unbind());
        }
        Ok(py.NotImplemented())
    }

    fn __rtruediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        // other / self
        self.ensure_multiplicative(&self.units)?;
        let arr = self.magnitude.bind(py);
        if let Ok(val) = other.extract::<f64>() {
            let result: Bound<'_, PyArray1<f64>> =
                arr.call_method1("__rtruediv__", (val,))?.downcast_into()?;
            let inv_units = {
                let dim = crate::units_container::UnitsContainer::new();
                &dim / &self.units
            };
            return self.wrap(py, result, inv_units);
        }
        if let Ok(q) = other.extract::<PyQuantity>() {
            self.ensure_multiplicative(&q.units)?;
            let result: Bound<'_, PyArray1<f64>> = arr
                .call_method1("__rtruediv__", (q.magnitude,))?
                .downcast_into()?;
            return self.wrap(py, result, &q.units / &self.units);
        }
        Ok(py.NotImplemented())
    }

    fn copy(&self, py: Python<'_>) -> PyResult<Self> {
        let arr = self.magnitude.bind(py);
        let copied: Bound<'_, PyArray1<f64>> = arr.call_method0("copy")?.downcast_into()?;
        Ok(self.rebuild(copied, self.units.clone()))
    }

    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let arr = self.magnitude.bind(py);
        let result = arr.call_method1("__getitem__", (key,))?;
        // If result is an ndarray, wrap as RustArrayQuantity; otherwise scalar Quantity
        if result.downcast::<PyArray1<f64>>().is_ok() {
            let new_arr: Bound<'_, PyArray1<f64>> = result.downcast_into()?;
            return self.wrap(py, new_arr, self.units.clone());
        }
        let val: f64 = result.extract()?;
        Ok(
            PyQuantity::from_parts(val, self.units.clone(), Arc::clone(&self.registry))
                .into_pyobject(py)?
                .into_any()
                .unbind(),
        )
    }

    fn __setitem__(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let arr = self.magnitude.bind(py);
        if let Ok(q) = value.extract::<PyQuantity>() {
            // Convert units, then assign scalar magnitude
            let mut reg = self.registry.lock().unwrap();
            let factor = reg.get_conversion_factor(&q.units, &self.units).to_py()?;
            drop(reg);
            let converted = q.magnitude * factor;
            arr.as_any().call_method1("__setitem__", (key, converted))?;
        } else if let Ok(aq) = value.extract::<PyRef<'_, Self>>() {
            // Convert units, then assign array magnitude
            let mag = if aq.units == self.units {
                aq.magnitude.bind(py).as_any().clone().unbind()
            } else {
                let mut reg = self.registry.lock().unwrap();
                let factor = reg.get_conversion_factor(&aq.units, &self.units).to_py()?;
                drop(reg);
                np_mul_scalar(aq.magnitude.bind(py), factor)?
                    .into_any()
                    .unbind()
            };
            arr.as_any().call_method1("__setitem__", (key, mag))?;
        } else {
            // Raw numeric value. If the quantity is dimensionless but stored in
            // scaled units (e.g. `ms/s`), interpret the bare value as a
            // dimensionless value and convert it into the stored unit before
            // assigning (issue #5); otherwise assign directly.
            let mut reg = self.registry.lock().unwrap();
            let dim = reg.get_dimensionality(&self.units).to_py()?;
            let factor = if dim.is_empty() {
                reg.get_conversion_factor(&UnitsContainer::new(), &self.units)
                    .to_py()?
            } else {
                1.0
            };
            drop(reg);
            if factor != 1.0 {
                let np = py.import("numpy")?;
                let value_arr = np.call_method1("asarray", (value,))?;
                let converted = value_arr.call_method1("__mul__", (factor,))?;
                arr.as_any().call_method1("__setitem__", (key, converted))?;
            } else {
                arr.as_any().call_method1("__setitem__", (key, value))?;
            }
        }
        Ok(())
    }

    fn __neg__(&self, py: Python<'_>) -> PyResult<Self> {
        let arr = self.magnitude.bind(py);
        let result: Bound<'_, PyArray1<f64>> = arr.call_method0("__neg__")?.downcast_into()?;
        Ok(self.rebuild(result, self.units.clone()))
    }

    fn __iter__(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<PyObject> {
        // Return an iterator that yields scalar Quantity objects
        let this = slf.borrow();
        let arr = this.magnitude.bind(py);
        let n = arr.len();

        let items: Vec<PyObject> = (0..n)
            .map(|i| {
                let v: f64 = arr
                    .call_method1("__getitem__", (i,))
                    .unwrap()
                    .extract()
                    .unwrap();
                PyQuantity::from_parts(v, this.units.clone(), Arc::clone(&this.registry))
                    .into_pyobject(py)
                    .unwrap()
                    .into_any()
                    .unbind()
            })
            .collect();

        let list = pyo3::types::PyList::new(py, items)?;
        list.call_method0("__iter__")?.extract()
    }

    fn reshape(&self, py: Python<'_>, shape: PyObject) -> PyResult<PyObject> {
        // reshape may return a non-1D array, so return as generic PyObject
        let arr = self.magnitude.bind(py);
        let reshaped = arr.call_method1("reshape", (shape,))?;
        // Wrap in a Python-level ArrayQuantity for full feature support
        let py_aq_cls = py
            .import("pintrs.numpy_support")?
            .getattr("ArrayQuantity")?;
        let reg = self.registry.lock().unwrap();
        let units_str = reg.format_units(&self.units);
        drop(reg);
        py_aq_cls
            .call1((reshaped, units_str, self._registry(py)?))?
            .extract()
    }

    fn is_compatible_with(&self, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let other_str: String = if let Ok(s) = other.extract::<String>() {
            s
        } else if let Ok(u) = other.extract::<PyUnit>() {
            let reg = u.registry.lock().unwrap();
            reg.format_units(&u.units)
        } else {
            return Ok(false);
        };
        let mut reg = self.registry.lock().unwrap();
        let self_dim = reg.get_dimensionality(&self.units).to_py()?;
        let other_uc = reg.parse_unit_expr(&other_str).to_py()?;
        let other_dim = reg.get_dimensionality(&other_uc).to_py()?;
        Ok(self_dim == other_dim)
    }
}
