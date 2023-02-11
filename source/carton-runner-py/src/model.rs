use std::{collections::HashMap, sync::atomic::AtomicU64};

use carton_runner_interface::{
    server::SealHandle,
    types::{Tensor, TensorStorage},
};
use numpy::{PyArrayDyn, ToPyArray};
use pyo3::{FromPyObject, PyAny, PyObject, PyResult, Python, ToPyObject};

enum SealImpl {
    /// Seal implemented in python
    Py(PyObject),

    /// Just store tensors
    Store {
        data: HashMap<SealHandle, HashMap<String, PyObject>>,
        counter: AtomicU64,
    },
}

/// A model can implement either or both of the following in python:
/// - `infer_with_tensors`
/// - `infer_with_handle` and `seal`
///
/// If both sets of methods are implemented, we just pass through to python for everything
/// If only `infer_with_tensors` is implemented, we use our own seal implementation that just stores tensors in a map
/// If only `infer_with_handle` and `seal` are implemented, we implement `infer_with_tensors` on top of those two
pub(crate) struct Model {
    // We store these to ensure that the relevant files don't get deleted while the model still exists
    _model_dir: tempfile::TempDir,
    _temp_packages: tempfile::TempDir,

    // The model returned from the entrypoint
    _model: PyObject,

    seal: SealImpl,
    infer_with_tensors: Option<PyObject>,
    infer_with_handle: Option<PyObject>,
}

impl Model {
    pub fn new(
        model_dir: tempfile::TempDir,
        temp_packages: tempfile::TempDir,
        model: &PyAny,
    ) -> Self {
        // Get methods from the model
        let seal = model
            .getattr(pyo3::intern!(model.py(), "seal"))
            .map(|item| item.into())
            .ok();
        let infer_with_tensors = model
            .getattr(pyo3::intern!(model.py(), "infer_with_tensors"))
            .map(|item| item.into())
            .ok();
        let infer_with_handle = model
            .getattr(pyo3::intern!(model.py(), "infer_with_handle"))
            .map(|item| item.into())
            .ok();

        if infer_with_tensors.is_some() || (infer_with_handle.is_some() && seal.is_some()) {
            // This is fine
        } else {
            panic!("Invalid set of methods implemented on this model! Must have `infer_with_tensors` and/or (`seal` and `infer_with_handle`)")
        }

        Self {
            _model_dir: model_dir,
            _temp_packages: temp_packages,
            _model: model.into(),
            seal: match seal {
                Some(item) => SealImpl::Py(item),
                None => SealImpl::Store {
                    data: Default::default(),
                    counter: Default::default(),
                },
            },
            infer_with_tensors,
            infer_with_handle,
        }
    }

    pub fn seal(&mut self, tensors: HashMap<String, Tensor>) -> PyResult<SealHandle> {
        // Convert to numpy arrays
        let tensors = to_numpy_arrays(tensors);

        match &mut self.seal {
            SealImpl::Py(seal) => Python::with_gil(|py| {
                let handle: u64 = seal.call1(py, (tensors,))?.extract(py)?;
                Ok(SealHandle::new(handle))
            }),
            SealImpl::Store { data, counter } => {
                let handle =
                    SealHandle::new(counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
                data.insert(handle, tensors);
                Ok(handle)
            }
        }
    }

    pub fn infer_with_handle(&mut self, handle: SealHandle) -> PyResult<HashMap<String, Tensor>> {
        match &mut self.seal {
            SealImpl::Py(_) => {
                // Seal being implemented implies that infer_with_handle is also implemented (as we checked in `new`)
                let infer_with_handle = self.infer_with_handle.as_ref().unwrap();
                Python::with_gil(|py| {
                    infer_with_handle
                        .call1(py, (handle.get(),))?
                        .extract(py)
                        .map(|item: HashMap<String, PythonTensorType>| item.convert())
                })
            }
            SealImpl::Store { data, .. } => {
                // TODO: return an error instead of using unwrap
                let tensors = data.remove(&handle).unwrap();

                // Run inference with tensors
                Python::with_gil(|py| {
                    self.infer_with_tensors
                        .as_ref()
                        .unwrap()
                        .call1(py, (tensors,))?
                        .extract(py)
                        .map(|item: HashMap<String, PythonTensorType>| item.convert())
                })
            }
        }
    }

    pub fn infer_with_tensors(
        &self,
        tensors: HashMap<String, Tensor>,
    ) -> PyResult<HashMap<String, Tensor>> {
        // Convert to numpy arrays
        let tensors = to_numpy_arrays(tensors);

        match &self.infer_with_tensors {
            Some(infer_with_tensors) => {
                // Run inference with tensors
                Python::with_gil(|py| {
                    infer_with_tensors
                        .call1(py, (tensors,))?
                        .extract(py)
                        .map(|item: HashMap<String, PythonTensorType>| item.convert())
                })
            }
            None => {
                // Implement on top of `seal` and `infer_with_handle`
                if let SealImpl::Py(seal) = &self.seal {
                    Python::with_gil(|py| {
                        let handle: u64 = seal.call1(py, (tensors,))?.extract(py)?;

                        // Seal being implemented implies that infer_with_handle is also implemented (as we checked in `new`)
                        let infer_with_handle = self.infer_with_handle.as_ref().unwrap();

                        infer_with_handle
                            .call1(py, (handle,))?
                            .extract(py)
                            .map(|item: HashMap<String, PythonTensorType>| item.convert())
                    })
                } else {
                    panic!("`infer_with_tensors` wasn't implemented and `seal` wasn't implemented either");
                }
            }
        }
    }
}

#[derive(FromPyObject)]
enum PythonTensorType<'py> {
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

trait ToTensorMap {
    fn convert(self) -> HashMap<String, Tensor>;
}

impl ToTensorMap for HashMap<String, PythonTensorType<'_>> {
    fn convert(self) -> HashMap<String, Tensor> {
        self.into_iter().map(|(k, v)| (k, v.into())).collect()
    }
}

/// Convert a numpy array to a tensor
/// TODO: this makes a copy
unsafe fn convert_tensor<'a, T: numpy::Element>(tensor: &'a PyArrayDyn<T>) -> Tensor
where
    TensorStorage<T>: From<ndarray::ArrayViewD<'a, T>>,
    Tensor: From<TensorStorage<T>>,
{
    let storage: TensorStorage<_> = tensor.as_array().into();
    storage.into()
}

impl From<PythonTensorType<'_>> for Tensor {
    fn from(value: PythonTensorType) -> Self {
        unsafe {
            match value {
                PythonTensorType::Float(item) => convert_tensor(item),
                PythonTensorType::Double(item) => convert_tensor(item),
                PythonTensorType::I8(item) => convert_tensor(item),
                PythonTensorType::I16(item) => convert_tensor(item),
                PythonTensorType::I32(item) => convert_tensor(item),
                PythonTensorType::I64(item) => convert_tensor(item),
                PythonTensorType::U8(item) => convert_tensor(item),
                PythonTensorType::U16(item) => convert_tensor(item),
                PythonTensorType::U32(item) => convert_tensor(item),
                PythonTensorType::U64(item) => convert_tensor(item),
            }
        }
    }
}

/// Convert a map of tensors to a map of numpy arrays
fn to_numpy_arrays(tensors: HashMap<String, Tensor>) -> HashMap<String, pyo3::PyObject> {
    Python::with_gil(|py| {
        tensors
            .into_iter()
            .map(|(k, v)| {
                // TODO: all of these make copies
                let transformed = match v {
                    Tensor::Float(item) => item.view().to_pyarray(py).to_object(py),
                    Tensor::Double(item) => item.view().to_pyarray(py).to_object(py),
                    // Tensor::String(item) => item.view().to_pyarray(py).to_object(py),
                    Tensor::String(_) => {
                        panic!("String tensors not implemented yet in the python runner")
                    }
                    Tensor::I8(item) => item.view().to_pyarray(py).to_object(py),
                    Tensor::I16(item) => item.view().to_pyarray(py).to_object(py),
                    Tensor::I32(item) => item.view().to_pyarray(py).to_object(py),
                    Tensor::I64(item) => item.view().to_pyarray(py).to_object(py),
                    Tensor::U8(item) => item.view().to_pyarray(py).to_object(py),
                    Tensor::U16(item) => item.view().to_pyarray(py).to_object(py),
                    Tensor::U32(item) => item.view().to_pyarray(py).to_object(py),
                    Tensor::U64(item) => item.view().to_pyarray(py).to_object(py),
                    Tensor::NestedTensor(_) => {
                        panic!("Nested tensors not implemented yet in the python runner")
                    }
                };

                (k, transformed)
            })
            .collect()
    })
}
