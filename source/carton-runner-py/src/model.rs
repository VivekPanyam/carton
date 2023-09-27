// Copyright 2023 Vivek Panyam
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::{collections::HashMap, sync::atomic::AtomicU64};

use carton_runner_interface::{
    server::SealHandle,
    types::{Tensor, TensorStorage},
};
use carton_utils_py::tensor::PyStringArrayType;
use futures_util::StreamExt;
use numpy::{PyArrayDyn, ToPyArray};
use pyo3::{FromPyObject, PyAny, PyErr, PyObject, Python, ToPyObject};

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

    pub fn seal(&mut self, tensors: HashMap<String, Tensor>) -> Result<SealHandle, String> {
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
        .map_err(pyerr_to_string_with_traceback)
    }

    pub async fn infer_with_handle(
        &mut self,
        handle: SealHandle,
    ) -> Result<impl futures::Stream<Item = Result<HashMap<String, Tensor>, String>>, String> {
        match &mut self.seal {
            SealImpl::Py(_) => {
                // Seal being implemented implies that infer_with_handle is also implemented (as we checked in `new`)
                let infer_with_handle = self.infer_with_handle.as_ref().unwrap();
                let res = Python::with_gil(|py| infer_with_handle.call1(py, (handle.get(),)));

                process_infer_output(res).await
            }
            SealImpl::Store { data, .. } => {
                // TODO: return an error instead of using unwrap
                let tensors = data.remove(&handle).unwrap();

                // Run inference with tensors
                let res = Python::with_gil(|py| {
                    self.infer_with_tensors
                        .as_ref()
                        .unwrap()
                        .call1(py, (tensors,))
                });

                process_infer_output(res).await
            }
        }
        .map_err(pyerr_to_string_with_traceback)
    }

    pub async fn infer_with_tensors(
        &self,
        tensors: HashMap<String, Tensor>,
    ) -> Result<impl futures::Stream<Item = Result<HashMap<String, Tensor>, String>>, String> {
        // Convert to numpy arrays
        let tensors = to_numpy_arrays(tensors);

        match &self.infer_with_tensors {
            Some(infer_with_tensors) => {
                // Run inference with tensors
                let res = Python::with_gil(|py| {
                    infer_with_tensors
                        .call1(py, (tensors,))
                });

                process_infer_output(res).await
            }
            None => {
                // Implement on top of `seal` and `infer_with_handle`
                if let SealImpl::Py(seal) = &self.seal {
                    let res = Python::with_gil(|py| {
                        let handle: u64 = seal.call1(py, (tensors,))?.extract(py)?;

                        // Seal being implemented implies that infer_with_handle is also implemented (as we checked in `new`)
                        let infer_with_handle = self.infer_with_handle.as_ref().unwrap();

                        infer_with_handle
                            .call1(py, (handle,))
                    });

                    process_infer_output(res).await
                } else {
                    panic!("`infer_with_tensors` wasn't implemented and `seal` wasn't implemented either");
                }
            }
        }.map_err(pyerr_to_string_with_traceback)
    }
}

pub(crate) fn pyerr_to_string_with_traceback(e: PyErr) -> String {
    let error_value = e.to_string();
    let traceback = Python::with_gil(|py| e.traceback(py).map(|t| t.format().unwrap()));

    format!("{}\n{}", error_value, traceback.unwrap_or_default())
}

#[derive(FromPyObject)]
enum PythonTensorType<'py> {
    Float(&'py PyArrayDyn<f32>),
    Double(&'py PyArrayDyn<f64>),

    I8(&'py PyArrayDyn<i8>),
    I16(&'py PyArrayDyn<i16>),
    I32(&'py PyArrayDyn<i32>),
    I64(&'py PyArrayDyn<i64>),

    U8(&'py PyArrayDyn<u8>),
    U16(&'py PyArrayDyn<u16>),
    U32(&'py PyArrayDyn<u32>),
    U64(&'py PyArrayDyn<u64>),

    String(PyStringArrayType<'py>),
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
                PythonTensorType::String(item) => {
                    // TODO: this makes two copies... (one in to_ndarray and one in view().into())
                    let arr = item.to_ndarray();
                    let storage: TensorStorage<String> = arr.view().into();
                    storage.into()
                }
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
                    Tensor::String(item) => PyStringArrayType::from_ndarray(py, item.view()),
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

/// Takes the output of an infer call and handles async iteration responses correctly
async fn process_infer_output(
    res: pyo3::PyResult<pyo3::Py<PyAny>>,
) -> pyo3::PyResult<impl futures::Stream<Item = Result<HashMap<String, Tensor>, String>>> {
    let res = res?;

    // Check if it's just a dictionary
    let dict = Python::with_gil(|py| {
        // Try and extract a dictionary
        res.extract(py)
            .map(|item: HashMap<String, PythonTensorType>| item.convert())
            .map_err(pyerr_to_string_with_traceback)
    });

    let stream = async_stream::try_stream! {
        if dict.is_ok() {
            // We're returning a single response
            yield dict?;
        } else {
            // Treat the response as an async iterator
            let mut rx = Python::with_gil(|py| pyo3_asyncio::tokio::into_stream_v2(res.as_ref(py)).map_err(pyerr_to_string_with_traceback))?;
            while let Some(item) = rx.next().await {
                let dict = Python::with_gil(|py| {
                    // Try and extract a dictionary
                    item.extract(py)
                        .map(|item: HashMap<String, PythonTensorType>| item.convert())
                        .map_err(pyerr_to_string_with_traceback)
                });

                yield dict?;
            }
        }
    };

    Ok(stream)
}
