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

use carton_core::types::{Tensor, TypedStorage};
use carton_utils_py::tensor::PyStringArrayType;
use ndarray::ShapeBuilder;
use numpy::{PyArrayDyn, ToPyArray};
use pyo3::{FromPyObject, Py, PyObject, Python, ToPyObject};

#[derive(FromPyObject)]
pub(crate) enum SupportedTensorType<'py> {
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

pub(crate) struct TypedPyTensorStorage<T> {
    /// This keeps the data "alive" while this tensor is in scope
    _keepalive: Py<PyArrayDyn<T>>,
    ptr: *const T,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

/// See the note in the `TypedStorage` impl below for why this is safe
unsafe impl<T> Send for TypedPyTensorStorage<T> where T: Send {}
unsafe impl<T> Sync for TypedPyTensorStorage<T> where T: Sync {}

impl<T: numpy::Element + std::fmt::Debug> TypedPyTensorStorage<T> {
    fn new(item: &PyArrayDyn<T>) -> Self {
        // We don't want to hold the GIL in the methods in `TypedStorage<T>`. This means
        // we can't do any operations on PyArrayDyn. Therefore, we extract the shape and
        // pointer below. See the safety notes in the TypedStorage impl below.

        let view = unsafe { item.as_array() };
        let ptr = view.as_ptr();
        let shape = view.shape();
        let strides = view.strides();

        for item in strides {
            if item < &0 {
                // TODO: don't panic; return an error
                panic!("Invalid strides when wrapping numpy array: {item}");
            }
        }

        Self {
            _keepalive: item.into(),
            ptr,
            shape: shape.into(),
            strides: strides.into_iter().map(|item| *item as usize).collect(),
        }
    }
}

impl<T> TypedStorage<T> for TypedPyTensorStorage<T> {
    fn view(&self) -> ndarray::ArrayViewD<T> {
        // SAFETY: Because we hold Py<_> of the array in `_keepalive`, the array has not been
        // reclaimed or deallocated by python.
        // TODO: confirm that there isn't a way for the shape or data pointer to change
        unsafe {
            ndarray::ArrayViewD::from_shape_ptr(
                self.shape.clone().strides(self.strides.clone()),
                self.ptr,
            )
        }
    }

    fn view_mut(&mut self) -> ndarray::ArrayViewMutD<T> {
        // SAFETY: Because we hold Py<_> of the array in `_keepalive`, the array has not been
        // reclaimed or deallocated by python.
        // TODO: confirm that there isn't a way for the shape or data pointer to change
        unsafe {
            ndarray::ArrayViewMutD::from_shape_ptr(
                self.shape.clone().strides(self.strides.clone()),
                self.ptr as _,
            )
        }
    }
}

impl<T: numpy::Element + std::fmt::Debug> From<&PyArrayDyn<T>> for TypedPyTensorStorage<T> {
    fn from(value: &PyArrayDyn<T>) -> Self {
        Self::new(value)
    }
}

pub struct StringPyTensorStorage {
    inner: ndarray::ArrayD<String>,
}

impl StringPyTensorStorage {
    fn new(item: ndarray::ArrayD<String>) -> Self {
        Self { inner: item }
    }
}

impl TypedStorage<String> for StringPyTensorStorage {
    fn view(&self) -> ndarray::ArrayViewD<String> {
        self.inner.view()
    }

    fn view_mut(&mut self) -> ndarray::ArrayViewMutD<String> {
        self.inner.view_mut()
    }
}

impl From<SupportedTensorType<'_>> for Tensor {
    fn from(value: SupportedTensorType<'_>) -> Self {
        match value {
            SupportedTensorType::Float(item) => {
                Tensor::Float(TypedPyTensorStorage::from(item).into())
            }
            SupportedTensorType::Double(item) => {
                Tensor::Double(TypedPyTensorStorage::from(item).into())
            }
            SupportedTensorType::String(item) => {
                let arr = item.to_ndarray();
                let out = StringPyTensorStorage::new(arr);
                Tensor::String(out.into())
            }

            SupportedTensorType::I8(item) => Tensor::I8(TypedPyTensorStorage::from(item).into()),
            SupportedTensorType::I16(item) => Tensor::I16(TypedPyTensorStorage::from(item).into()),
            SupportedTensorType::I32(item) => Tensor::I32(TypedPyTensorStorage::from(item).into()),
            SupportedTensorType::I64(item) => Tensor::I64(TypedPyTensorStorage::from(item).into()),

            SupportedTensorType::U8(item) => Tensor::U8(TypedPyTensorStorage::from(item).into()),
            SupportedTensorType::U16(item) => Tensor::U16(TypedPyTensorStorage::from(item).into()),
            SupportedTensorType::U32(item) => Tensor::U32(TypedPyTensorStorage::from(item).into()),
            SupportedTensorType::U64(item) => Tensor::U64(TypedPyTensorStorage::from(item).into()),
        }
    }
}

pub(crate) fn tensor_to_py(item: &Tensor) -> PyObject {
    // TODO this makes a copy
    Python::with_gil(|py| {
        match item {
            Tensor::Float(item) => item.view().to_pyarray(py).to_object(py),
            Tensor::Double(item) => item.view().to_pyarray(py).to_object(py),
            Tensor::String(item) => {
                // Strings are a bit more complex and require conversion
                // This is unfortunate but necessary because most frameworks have different internal representations of strings
                let view = item.view();
                PyStringArrayType::from_ndarray(py, view)
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
                panic!("Nested tensor output not implemented yet")
            }
        }
    })
}
