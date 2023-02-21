use carton_core::types::{Tensor, TensorStorage, TypedStorage};
use ndarray::ShapeBuilder;
use numpy::{PyArrayDyn, ToPyArray};
use pyo3::{FromPyObject, Py, PyObject, Python, ToPyObject};

#[derive(FromPyObject)]
pub(crate) enum SupportedTensorType<'py> {
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

pub(crate) struct PyTensorStorage {}

impl TensorStorage for PyTensorStorage {
    type TypedStorage<T> = TypedPyTensorStorage<T>
    where
        T: Send + Sync;
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

pub(crate) fn tensor_to_py<T: TensorStorage>(item: &Tensor<T>) -> PyObject {
    // TODO this makes a copy
    Python::with_gil(|py| {
        match item {
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
    })
}
