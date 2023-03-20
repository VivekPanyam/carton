use std::{collections::HashMap, str::FromStr, sync::Arc};

use carton_core::{
    info::RunnerInfo,
    types::{
        Device, GenericStorage, LoadOpts, PackOpts, RunnerOpt, Tensor, TensorStorage, TypedStorage,
    },
};
use ndarray::ShapeBuilder;
use numpy::{PyArrayDyn, ToPyArray};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use semver::VersionReq;

#[derive(FromPyObject)]
enum SupportedTensorType<'py> {
    Float(&'py PyArrayDyn<f32>),
    Double(&'py PyArrayDyn<f64>),
    // TODO: handle this
    // String(&'py PyArrayDyn<PyString>),
    I8(&'py PyArrayDyn<i8>),
    I16(&'py PyArrayDyn<i16>),
    I32(&'py PyArrayDyn<i32>),
    I64(&'py PyArrayDyn<i64>),

    U8(&'py PyArrayDyn<u8>),
    U16(&'py PyArrayDyn<u16>),
    U32(&'py PyArrayDyn<u32>),
    U64(&'py PyArrayDyn<u64>),
}

struct PyTensorStorage {}

impl TensorStorage for PyTensorStorage {
    type TypedStorage<T> = TypedPyTensorStorage<T>
    where
        T: Send + Sync;
}

struct TypedPyTensorStorage<T> {
    /// This keeps the data "alive" while this tensor is in scope
    _keepalive: Py<PyArrayDyn<T>>,
    shape: ndarray::StrideShape<ndarray::IxDyn>,
    ptr: *mut T,
}

/// See the note in the `TypedStorage` impl below for why this is safe
unsafe impl<T> Send for TypedPyTensorStorage<T> where T: Send {}
unsafe impl<T> Sync for TypedPyTensorStorage<T> where T: Sync {}

impl<T: numpy::Element> TypedPyTensorStorage<T> {
    fn new(item: &PyArrayDyn<T>) -> Self {
        // We don't want to hold the GIL in the methods in `TypedStorage<T>`. This means
        // we can't do any operations on PyArrayDyn. Therefore, we extract the shape and
        // pointer below. See the safety notes in the TypedStorage impl below.

        let shape = item.shape().strides(
            &item
                .strides()
                .into_iter()
                .map(|s| (*s).try_into().unwrap())
                .collect::<Vec<usize>>(),
        );

        let ptr = item.data();

        Self {
            _keepalive: item.into(),
            shape,
            ptr,
        }
    }
}

impl<T> TypedStorage<T> for TypedPyTensorStorage<T> {
    fn view(&self) -> ndarray::ArrayViewD<T> {
        // SAFETY: Because we hold Py<_> of the array in `_keepalive`, the array has not been
        // reclaimed or deallocated by python.
        // TODO: confirm that there isn't a way for the shape or data pointer to change
        unsafe { ndarray::ArrayViewD::from_shape_ptr(self.shape.clone(), self.ptr) }
    }

    fn view_mut(&mut self) -> ndarray::ArrayViewMutD<T> {
        // SAFETY: Because we hold Py<_> of the array in `_keepalive`, the array has not been
        // reclaimed or deallocated by python.
        // TODO: confirm that there isn't a way for the shape or data pointer to change
        unsafe { ndarray::ArrayViewMutD::from_shape_ptr(self.shape.clone(), self.ptr as _) }
    }
}

impl<T: numpy::Element> From<&PyArrayDyn<T>> for TypedPyTensorStorage<T> {
    fn from(value: &PyArrayDyn<T>) -> Self {
        Self::new(value)
    }
}

impl From<SupportedTensorType<'_>> for Tensor<PyTensorStorage> {
    fn from(value: SupportedTensorType<'_>) -> Self {
        match value {
            SupportedTensorType::Float(item) => Tensor::Float(item.into()),
            SupportedTensorType::Double(item) => Tensor::Double(item.into()),

            SupportedTensorType::I8(item) => Tensor::I8(item.into()),
            SupportedTensorType::I16(item) => Tensor::I16(item.into()),
            SupportedTensorType::I32(item) => Tensor::I32(item.into()),
            SupportedTensorType::I64(item) => Tensor::I64(item.into()),

            SupportedTensorType::U8(item) => Tensor::U8(item.into()),
            SupportedTensorType::U16(item) => Tensor::U16(item.into()),
            SupportedTensorType::U32(item) => Tensor::U32(item.into()),
            SupportedTensorType::U64(item) => Tensor::U64(item.into()),
        }
    }
}

#[pyclass]
#[derive(Clone)]
struct SealHandle {
    inner: carton_core::types::SealHandle,
}

#[pyclass]
struct Carton {
    inner: Arc<carton_core::Carton>,
}
// TODO do we need with_gil?

#[pymethods]
impl Carton {
    #[getter]
    fn name(&self) -> PyResult<Option<&String>> {
        Ok(self.inner.get_info().model_name.as_ref())
    }

    #[getter]
    fn runner(&self) -> PyResult<&str> {
        Ok(&self.inner.get_info().runner.runner_name)
    }

    fn seal<'a>(&self, py: Python<'a>, tensors: &PyDict) -> PyResult<&'a PyAny> {
        let tensors: HashMap<String, SupportedTensorType> = tensors.extract().unwrap();
        let transformed = tensors.into_iter().map(|(k, v)| (k, v.into())).collect();

        let inner = self.inner.clone();
        pyo3_asyncio::tokio::future_into_py(py, async move {
            let out = inner.seal(transformed).await.unwrap();
            Ok(SealHandle { inner: out })
        })
    }

    // TODO: merge the infer methods into one
    fn infer_with_inputs<'a>(&self, py: Python<'a>, tensors: &PyDict) -> PyResult<&'a PyAny> {
        let tensors: HashMap<String, SupportedTensorType> = tensors.extract().unwrap();
        let transformed = tensors.into_iter().map(|(k, v)| (k, v.into())).collect();

        let inner = self.inner.clone();
        pyo3_asyncio::tokio::future_into_py(py, async move {
            let out = inner.infer_with_inputs(transformed).await.unwrap();

            let mut transformed: HashMap<String, PyObject> = HashMap::new();

            for (k, v) in out {
                // TODO this makes a copy
                let pytype = Python::with_gil(|py| {
                    match v {
                        Tensor::Float(item) => item.view().to_pyarray(py).to_object(py),
                        Tensor::Double(item) => item.view().to_pyarray(py).to_object(py),
                        // Tensor::String(item) => item.view().to_pyarray(py).to_object(py),
                        Tensor::String(_) => panic!("String tensor output not implemented yet"),
                        Tensor::I8(item) => item.view().to_pyarray(py).to_object(py),
                        Tensor::I16(item) => item.view().to_pyarray(py).to_object(py),
                        Tensor::I32(item) => item.view().to_pyarray(py).to_object(py),
                        Tensor::I64(item) => item.view().to_pyarray(py).to_object(py),
                        Tensor::U8(item) => item.view().to_pyarray(py).to_object(py),
                        Tensor::U16(item) => item.view().to_pyarray(py).to_object(py),
                        Tensor::U32(item) => item.view().to_pyarray(py).to_object(py),
                        Tensor::U64(item) => item.view().to_pyarray(py).to_object(py),
                        Tensor::NestedTensor(_) => {
                            panic!("Nested tensor output not implemented yet")
                        }
                    }
                });

                transformed.insert(k, pytype);
            }

            Ok(transformed)
        })
    }

    // TODO: merge the infer methods into one
    fn infer_with_handle<'a>(&self, py: Python<'a>, handle: SealHandle) -> PyResult<&'a PyAny> {
        let inner = self.inner.clone();
        pyo3_asyncio::tokio::future_into_py(py, async move {
            let out = inner.infer_with_handle(handle.inner).await.unwrap();

            let mut transformed: HashMap<String, PyObject> = HashMap::new();

            for (k, v) in out {
                // TODO this makes a copy
                let pytype = Python::with_gil(|py| {
                    match v {
                        Tensor::Float(item) => item.view().to_pyarray(py).to_object(py),
                        Tensor::Double(item) => item.view().to_pyarray(py).to_object(py),
                        // Tensor::String(item) => item.view().to_pyarray(py).to_object(py),
                        Tensor::String(_) => panic!("String tensor output not implemented yet"),
                        Tensor::I8(item) => item.view().to_pyarray(py).to_object(py),
                        Tensor::I16(item) => item.view().to_pyarray(py).to_object(py),
                        Tensor::I32(item) => item.view().to_pyarray(py).to_object(py),
                        Tensor::I64(item) => item.view().to_pyarray(py).to_object(py),
                        Tensor::U8(item) => item.view().to_pyarray(py).to_object(py),
                        Tensor::U16(item) => item.view().to_pyarray(py).to_object(py),
                        Tensor::U32(item) => item.view().to_pyarray(py).to_object(py),
                        Tensor::U64(item) => item.view().to_pyarray(py).to_object(py),
                        Tensor::NestedTensor(_) => {
                            panic!("Nested tensor output not implemented yet")
                        }
                    }
                });

                transformed.insert(k, pytype);
            }

            Ok(transformed)
        })
    }
}

#[derive(FromPyObject)]
enum PyRunnerOpt {
    Integer(i64),
    Double(f64),
    String(String),
    Boolean(bool),
    // TODO: datetime
    // Date(DateTime<Utc>),
}

impl From<PyRunnerOpt> for RunnerOpt {
    fn from(value: PyRunnerOpt) -> Self {
        match value {
            PyRunnerOpt::Integer(v) => Self::Integer(v),
            PyRunnerOpt::Double(v) => Self::Double(v),
            PyRunnerOpt::String(v) => Self::String(v),
            PyRunnerOpt::Boolean(v) => Self::Boolean(v),
        }
    }
}

/// Loads a model
#[pyfunction]
fn load(
    py: Python,
    path: String,
    override_runner_name: Option<String>,
    override_required_framework_version: Option<String>,
    override_runner_opts: Option<HashMap<String, PyRunnerOpt>>,
    visible_device: String,
) -> PyResult<&PyAny> {
    pyo3_asyncio::tokio::future_into_py(py, async move {
        let opts = LoadOpts {
            override_runner_name,
            override_required_framework_version,
            override_runner_opts: override_runner_opts
                .map(|opts| opts.into_iter().map(|(k, v)| (k, v.into())).collect()),
            // TODO: use something more specific than ValueError
            visible_device: Device::maybe_from_str(&visible_device)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        };

        // TODO: use something more specific than ValueError
        let inner = carton_core::Carton::load(path, opts)
            .await
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Carton {
            inner: Arc::new(inner),
        })
    })
}

/// Load an unpacked model
#[pyfunction]
fn load_unpacked(
    py: Python,
    path: String,
    runner_name: String,
    required_framework_version: String,
    opts: Option<HashMap<String, PyRunnerOpt>>,
    visible_device: String,
) -> PyResult<&PyAny> {
    pyo3_asyncio::tokio::future_into_py(py, async move {
        let pack_opts = PackOpts::<GenericStorage> {
            model_name: None,
            short_description: None,
            model_description: None,
            required_platforms: None,
            inputs: None,
            outputs: None,
            self_tests: None,
            examples: None,
            runner: RunnerInfo {
                runner_name,
                required_framework_version: VersionReq::from_str(&required_framework_version)
                    .map_err(|e| {
                        PyValueError::new_err(format!("Invalid `required_framework_version`: {e}"))
                    })?,
                runner_compat_version: None,
                opts: opts.map(|opts| opts.into_iter().map(|(k, v)| (k, v.into())).collect()),
            },
            misc_files: None,
        };

        // TODO: use something more specific than ValueError
        let load_opts = LoadOpts {
            visible_device: Device::maybe_from_str(&visible_device)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
            ..Default::default()
        };

        let inner = carton_core::Carton::load_unpacked(path, pack_opts, load_opts)
            .await
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(Carton {
            inner: Arc::new(inner),
        })
    })
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn cartonml(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(load_unpacked, m)?)?;
    m.add_class::<Carton>()?;
    Ok(())
}
