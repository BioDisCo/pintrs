//! Rust-backed ArrayQuantity for fast numpy array + unit operations.

use std::sync::{Arc, Mutex};

use numpy::{PyArray1, PyUntypedArrayMethods};
use once_cell::sync::OnceCell;
use pyo3::prelude::*;

use crate::registry::UnitRegistry as RustRegistry;
use crate::units_container::UnitsContainer;
use crate::{to_py_err, PyQuantity, PyUnit, PyUnitRegistry, SharedRegistry};

/// Helper: multiply a numpy array by a scalar, returning a new array.
fn np_mul_scalar<'py>(
    arr: &Bound<'py, PyArray1<f64>>,
    factor: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let result = arr.call_method1("__mul__", (factor,))?;
    result.downcast_into().map_err(Into::into)
}

/// A Quantity whose magnitude is a numpy f64 array.
#[pyclass(name = "RustArrayQuantity", module = "pintrs._core", subclass)]
pub struct PyArrayQuantity {
    magnitude: Py<PyArray1<f64>>,
    units: UnitsContainer,
    registry: SharedRegistry,
    cached_registry_obj: OnceCell<PyObject>,
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
            let reg = RustRegistry::new().map_err(to_py_err)?;
            (Arc::new(Mutex::new(reg)), OnceCell::new())
        };

        let uc = {
            let mut reg = shared.lock().unwrap();
            reg.parse_unit_expr(units).map_err(to_py_err)?
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
        let dim = reg.get_dimensionality(&self.units).map_err(to_py_err)?;
        Ok(dim.to_string())
    }

    #[getter]
    fn dimensionless(&self) -> PyResult<bool> {
        let mut reg = self.registry.lock().unwrap();
        let dim = reg.get_dimensionality(&self.units).map_err(to_py_err)?;
        Ok(dim.is_empty())
    }

    #[getter]
    fn _registry(&self, py: Python<'_>) -> PyResult<PyObject> {
        if let Some(cached) = self.cached_registry_obj.get() {
            return Ok(cached.clone_ref(py));
        }
        let reg = PyUnitRegistry {
            inner: Arc::clone(&self.registry),
            _init_kwargs: None,
        };
        let obj = reg.into_pyobject(py)?.into_any().unbind();
        let _ = self.cached_registry_obj.set(obj.clone_ref(py));
        Ok(obj)
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
        let dst = reg.parse_unit_expr(&dst_str).map_err(to_py_err)?;
        let factor = reg
            .get_conversion_factor(&self.units, &dst)
            .map_err(to_py_err)?;
        drop(reg);

        let result = np_mul_scalar(self.magnitude.bind(py), factor)?;
        Ok(Self {
            magnitude: result.unbind(),
            units: dst,
            registry: Arc::clone(&self.registry),
            cached_registry_obj: OnceCell::new(),
        })
    }

    fn m_as(&self, py: Python<'_>, units: &Bound<'_, PyAny>) -> PyResult<Py<PyArray1<f64>>> {
        let dst_str = crate::extract_units_string(units)?;
        let mut reg = self.registry.lock().unwrap();
        let dst = reg.parse_unit_expr(&dst_str).map_err(to_py_err)?;
        let factor = reg
            .get_conversion_factor(&self.units, &dst)
            .map_err(to_py_err)?;
        drop(reg);

        let result = np_mul_scalar(self.magnitude.bind(py), factor)?;
        Ok(result.unbind())
    }

    fn to_base_units(&self, py: Python<'_>) -> PyResult<Self> {
        let mut reg = self.registry.lock().unwrap();
        let (factor, root_units) = reg.get_root_units(&self.units).map_err(to_py_err)?;
        drop(reg);

        let result = np_mul_scalar(self.magnitude.bind(py), factor)?;
        Ok(Self {
            magnitude: result.unbind(),
            units: root_units,
            registry: Arc::clone(&self.registry),
            cached_registry_obj: OnceCell::new(),
        })
    }

    // --- Reductions ---

    fn sum(&self, py: Python<'_>) -> PyResult<PyQuantity> {
        let total: f64 = self.magnitude.bind(py).call_method0("sum")?.extract()?;
        Ok(PyQuantity {
            magnitude: total,
            units: self.units.clone(),
            registry: Arc::clone(&self.registry),
            cached_dim: OnceCell::new(),
            cached_registry_obj: OnceCell::new(),
        })
    }

    fn mean(&self, py: Python<'_>) -> PyResult<PyQuantity> {
        let mean: f64 = self.magnitude.bind(py).call_method0("mean")?.extract()?;
        Ok(PyQuantity {
            magnitude: mean,
            units: self.units.clone(),
            registry: Arc::clone(&self.registry),
            cached_dim: OnceCell::new(),
            cached_registry_obj: OnceCell::new(),
        })
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
        let arr = self.magnitude.bind(py);
        if let Ok(val) = other.extract::<f64>() {
            let result: Bound<'_, PyArray1<f64>> =
                arr.call_method1("__mul__", (val,))?.downcast_into()?;
            return Ok(Self {
                magnitude: result.unbind(),
                units: self.units.clone(),
                registry: Arc::clone(&self.registry),
                cached_registry_obj: OnceCell::new(),
            }
            .into_pyobject(py)?
            .into_any()
            .unbind());
        }
        if let Ok(other_aq) = other.extract::<PyRef<'_, Self>>() {
            let b = other_aq.magnitude.bind(py);
            let result: Bound<'_, PyArray1<f64>> =
                arr.call_method1("__mul__", (b,))?.downcast_into()?;
            return Ok(Self {
                magnitude: result.unbind(),
                units: &self.units * &other_aq.units,
                registry: Arc::clone(&self.registry),
                cached_registry_obj: OnceCell::new(),
            }
            .into_pyobject(py)?
            .into_any()
            .unbind());
        }
        if let Ok(q) = other.extract::<PyQuantity>() {
            let result: Bound<'_, PyArray1<f64>> = arr
                .call_method1("__mul__", (q.magnitude,))?
                .downcast_into()?;
            return Ok(Self {
                magnitude: result.unbind(),
                units: &self.units * &q.units,
                registry: Arc::clone(&self.registry),
                cached_registry_obj: OnceCell::new(),
            }
            .into_pyobject(py)?
            .into_any()
            .unbind());
        }
        Ok(py.NotImplemented())
    }

    fn __rmul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        self.__mul__(py, other)
    }

    fn __add__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        if let Ok(other_aq) = other.extract::<PyRef<'_, Self>>() {
            let a = self.magnitude.bind(py);
            let b = other_aq.magnitude.bind(py);

            let result: Bound<'_, PyArray1<f64>> = if self.units == other_aq.units {
                a.call_method1("__add__", (b,))?.downcast_into()?
            } else {
                let mut reg = self.registry.lock().unwrap();
                let factor = reg
                    .get_conversion_factor(&other_aq.units, &self.units)
                    .map_err(to_py_err)?;
                drop(reg);
                let b_scaled = np_mul_scalar(b, factor)?;
                a.call_method1("__add__", (&b_scaled,))?.downcast_into()?
            };
            return Ok(Self {
                magnitude: result.unbind(),
                units: self.units.clone(),
                registry: Arc::clone(&self.registry),
                cached_registry_obj: OnceCell::new(),
            }
            .into_pyobject(py)?
            .into_any()
            .unbind());
        }
        Ok(py.NotImplemented())
    }

    fn __radd__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        self.__add__(py, other)
    }

    fn __sub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        if let Ok(other_aq) = other.extract::<PyRef<'_, Self>>() {
            let a = self.magnitude.bind(py);
            let b = other_aq.magnitude.bind(py);

            let result: Bound<'_, PyArray1<f64>> = if self.units == other_aq.units {
                a.call_method1("__sub__", (b,))?.downcast_into()?
            } else {
                let mut reg = self.registry.lock().unwrap();
                let factor = reg
                    .get_conversion_factor(&other_aq.units, &self.units)
                    .map_err(to_py_err)?;
                drop(reg);
                let b_scaled = np_mul_scalar(b, factor)?;
                a.call_method1("__sub__", (&b_scaled,))?.downcast_into()?
            };
            return Ok(Self {
                magnitude: result.unbind(),
                units: self.units.clone(),
                registry: Arc::clone(&self.registry),
                cached_registry_obj: OnceCell::new(),
            }
            .into_pyobject(py)?
            .into_any()
            .unbind());
        }
        Ok(py.NotImplemented())
    }

    fn __truediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        let arr = self.magnitude.bind(py);
        if let Ok(val) = other.extract::<f64>() {
            if val == 0.0 {
                return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                    "division by zero",
                ));
            }
            let result: Bound<'_, PyArray1<f64>> =
                arr.call_method1("__truediv__", (val,))?.downcast_into()?;
            return Ok(Self {
                magnitude: result.unbind(),
                units: self.units.clone(),
                registry: Arc::clone(&self.registry),
                cached_registry_obj: OnceCell::new(),
            }
            .into_pyobject(py)?
            .into_any()
            .unbind());
        }
        if let Ok(other_aq) = other.extract::<PyRef<'_, Self>>() {
            let b = other_aq.magnitude.bind(py);
            let result: Bound<'_, PyArray1<f64>> =
                arr.call_method1("__truediv__", (b,))?.downcast_into()?;
            return Ok(Self {
                magnitude: result.unbind(),
                units: &self.units / &other_aq.units,
                registry: Arc::clone(&self.registry),
                cached_registry_obj: OnceCell::new(),
            }
            .into_pyobject(py)?
            .into_any()
            .unbind());
        }
        if let Ok(q) = other.extract::<PyQuantity>() {
            if q.magnitude == 0.0 {
                return Err(pyo3::exceptions::PyZeroDivisionError::new_err(
                    "division by zero",
                ));
            }
            let result: Bound<'_, PyArray1<f64>> = arr
                .call_method1("__truediv__", (q.magnitude,))?
                .downcast_into()?;
            return Ok(Self {
                magnitude: result.unbind(),
                units: &self.units / &q.units,
                registry: Arc::clone(&self.registry),
                cached_registry_obj: OnceCell::new(),
            }
            .into_pyobject(py)?
            .into_any()
            .unbind());
        }
        if let Ok(u) = other.extract::<PyUnit>() {
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

    fn __neg__(&self, py: Python<'_>) -> PyResult<Self> {
        let arr = self.magnitude.bind(py);
        let result: Bound<'_, PyArray1<f64>> = arr.call_method0("__neg__")?.downcast_into()?;
        Ok(Self {
            magnitude: result.unbind(),
            units: self.units.clone(),
            registry: Arc::clone(&self.registry),
            cached_registry_obj: OnceCell::new(),
        })
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
                PyQuantity {
                    magnitude: v,
                    units: this.units.clone(),
                    registry: Arc::clone(&this.registry),
                    cached_dim: OnceCell::new(),
                    cached_registry_obj: OnceCell::new(),
                }
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
        let self_dim = reg.get_dimensionality(&self.units).map_err(to_py_err)?;
        let other_uc = reg.parse_unit_expr(&other_str).map_err(to_py_err)?;
        let other_dim = reg.get_dimensionality(&other_uc).map_err(to_py_err)?;
        Ok(self_dim == other_dim)
    }
}
