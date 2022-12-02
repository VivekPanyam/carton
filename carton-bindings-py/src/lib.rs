use std::{collections::HashMap, sync::Arc};

use carton_core::types::{Tensor, SealHandle};
use numpy::{PyArrayDyn, ToPyArray};
use pyo3::{prelude::*, types::PyDict};

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

#[pyclass]
struct Carton {
    pub inner: Arc<carton_core::client::Carton>
}
// TODO do we need with_gil?

#[pymethods]
impl Carton {
    #[getter]
    fn name(&self) -> PyResult<&str> {
        Ok(&self.inner.model_name)
    }

    #[getter]
    fn runner(&self) -> PyResult<&str> {
        Ok(&self.inner.model_runner)
    }

    fn seal<'a>(&self, py: Python<'a>, tensors: &PyDict) -> PyResult<&'a PyAny> {
        let mut transformed = HashMap::new();
        let tensors: HashMap<String, SupportedTensorType> = tensors.extract().unwrap();

        for (k, v) in tensors {

            // TODO: this makes a copy
            let native = match v {
                SupportedTensorType::Float(item) => Tensor::Float(item.to_owned_array()),
                SupportedTensorType::Double(item) => Tensor::Double(item.to_owned_array()),

                SupportedTensorType::I8(item) => Tensor::I8(item.to_owned_array()),
                SupportedTensorType::I16(item) => Tensor::I16(item.to_owned_array()),
                SupportedTensorType::I32(item) => Tensor::I32(item.to_owned_array()),
                SupportedTensorType::I64(item) => Tensor::I64(item.to_owned_array()),

                SupportedTensorType::U8(item) => Tensor::U8(item.to_owned_array()),
                SupportedTensorType::U16(item) => Tensor::U16(item.to_owned_array()),
                SupportedTensorType::U32(item) => Tensor::U32(item.to_owned_array()),
                SupportedTensorType::U64(item) => Tensor::U64(item.to_owned_array()),
            };

            transformed.insert(
                k,
                native
            );
        }

        let inner = self.inner.clone();
        pyo3_asyncio::tokio::future_into_py(py, async move {
            Ok(inner.seal(transformed).await.unwrap())
        })
    }

    // TODO: merge the infer methods into one
    fn infer_with_inputs<'a>(&self, py: Python<'a>, tensors: &PyDict) -> PyResult<&'a PyAny> {
        let mut transformed = HashMap::new();
        let tensors: HashMap<String, SupportedTensorType> = tensors.extract().unwrap();

        for (k, v) in tensors {

            // TODO: this makes a copy
            let native = match v {
                SupportedTensorType::Float(item) => Tensor::Float(item.to_owned_array()),
                SupportedTensorType::Double(item) => Tensor::Double(item.to_owned_array()),

                SupportedTensorType::I8(item) => Tensor::I8(item.to_owned_array()),
                SupportedTensorType::I16(item) => Tensor::I16(item.to_owned_array()),
                SupportedTensorType::I32(item) => Tensor::I32(item.to_owned_array()),
                SupportedTensorType::I64(item) => Tensor::I64(item.to_owned_array()),

                SupportedTensorType::U8(item) => Tensor::U8(item.to_owned_array()),
                SupportedTensorType::U16(item) => Tensor::U16(item.to_owned_array()),
                SupportedTensorType::U32(item) => Tensor::U32(item.to_owned_array()),
                SupportedTensorType::U64(item) => Tensor::U64(item.to_owned_array()),
            };

            transformed.insert(
                k,
                native
            );
        }

        let inner = self.inner.clone();
        pyo3_asyncio::tokio::future_into_py(py, async move {
            let out = inner.infer_with_inputs(transformed).await.unwrap();

            let mut transformed: HashMap<String, PyObject> = HashMap::new();

            for (k, v) in out {
                // TODO this makes a copy
                let pytype = Python::with_gil(|py| {
                    match v {
                        Tensor::Float(item) => item.to_pyarray(py).to_object(py),
                        Tensor::Double(item) => item.to_pyarray(py).to_object(py),
                        // Tensor::String(item) => item.to_pyarray(py).to_object(py),
                        Tensor::String(_) => panic!("String tensor output not implemented yet"),
                        Tensor::I8(item) => item.to_pyarray(py).to_object(py),
                        Tensor::I16(item) => item.to_pyarray(py).to_object(py),
                        Tensor::I32(item) => item.to_pyarray(py).to_object(py),
                        Tensor::I64(item) => item.to_pyarray(py).to_object(py),
                        Tensor::U8(item) => item.to_pyarray(py).to_object(py),
                        Tensor::U16(item) => item.to_pyarray(py).to_object(py),
                        Tensor::U32(item) => item.to_pyarray(py).to_object(py),
                        Tensor::U64(item) => item.to_pyarray(py).to_object(py),
                    }
                });

                transformed.insert(
                    k,
                    pytype
                );
            }

            Ok(transformed)
        })
    }


    // TODO: merge the infer methods into one
    fn infer_with_handle<'a>(&self, py: Python<'a>, handle: SealHandle) -> PyResult<&'a PyAny> {
        let inner = self.inner.clone();
        pyo3_asyncio::tokio::future_into_py(py, async move {
            let out = inner.infer_with_handle(handle).await.unwrap();

            let mut transformed: HashMap<String, PyObject> = HashMap::new();

            for (k, v) in out {
                // TODO this makes a copy
                let pytype = Python::with_gil(|py| {
                    match v {
                        Tensor::Float(item) => item.to_pyarray(py).to_object(py),
                        Tensor::Double(item) => item.to_pyarray(py).to_object(py),
                        // Tensor::String(item) => item.to_pyarray(py).to_object(py),
                        Tensor::String(_) => panic!("String tensor output not implemented yet"),
                        Tensor::I8(item) => item.to_pyarray(py).to_object(py),
                        Tensor::I16(item) => item.to_pyarray(py).to_object(py),
                        Tensor::I32(item) => item.to_pyarray(py).to_object(py),
                        Tensor::I64(item) => item.to_pyarray(py).to_object(py),
                        Tensor::U8(item) => item.to_pyarray(py).to_object(py),
                        Tensor::U16(item) => item.to_pyarray(py).to_object(py),
                        Tensor::U32(item) => item.to_pyarray(py).to_object(py),
                        Tensor::U64(item) => item.to_pyarray(py).to_object(py),
                    }
                });

                transformed.insert(
                    k,
                    pytype
                );
            }

            Ok(transformed)
        })
    }

}

/// Loads a model
#[pyfunction]
fn load(py: Python, path: String, runner: Option<String>, runner_version: Option<String>, runner_opts: Option<String>, visible_device: String) -> PyResult<&PyAny> {
    pyo3_asyncio::tokio::future_into_py(py, async {

        let inner = carton_core::client::Carton::new(path, runner, runner_version, runner_opts, visible_device).await.unwrap();
        Ok(Carton { inner: Arc::new(inner) })
    })
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn carton(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_class::<Carton>()?;
    Ok(())
}