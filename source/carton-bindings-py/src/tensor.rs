use carton_core::types::{Tensor, TensorStorage, TypedStorage};
use ndarray::ShapeBuilder;
use numpy::{PyArrayDyn, ToPyArray};
use pyo3::{
    types::PyTuple, AsPyPointer, FromPyObject, Py, PyAny, PyDowncastError, PyObject, Python,
    ToPyObject,
};

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

    String(SomeArrayType<'py>),
}

pub(crate) struct SomeArrayType<'a> {
    inner: &'a PyAny,
}

impl<'py> FromPyObject<'py> for SomeArrayType<'py> {
    fn extract(ob: &'py PyAny) -> pyo3::PyResult<Self> {
        unsafe {
            if numpy::npyffi::PyArray_Check(ob.py(), ob.as_ptr()) == 0 {
                return Err(PyDowncastError::new(ob, "SomeNumpyArray").into());
            }
        }

        Ok(SomeArrayType { inner: ob })
    }
}

pub(crate) struct PyTensorStorage {}

impl TensorStorage for PyTensorStorage {
    type TypedStorage<T> = TypedPyTensorStorage<T>
    where
        T: Send + Sync;

    type TypedStringStorage = StringPyTensorStorage;
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

impl From<SupportedTensorType<'_>> for Tensor<PyTensorStorage> {
    fn from(value: SupportedTensorType<'_>) -> Self {
        match value {
            SupportedTensorType::Float(item) => Tensor::Float(item.into()),
            SupportedTensorType::Double(item) => Tensor::Double(item.into()),
            SupportedTensorType::String(item) => {
                let item = item.inner;

                // Strings are a bit complex and require conversion
                // This is unfortunate but necessary because most frameworks have different internal representations of strings
                let kind: char = item
                    .getattr("dtype")
                    .unwrap()
                    .getattr("kind")
                    .unwrap()
                    .extract()
                    .unwrap();
                let itemsize: usize = item.getattr("itemsize").unwrap().extract().unwrap();
                if kind != 'U' {
                    panic!(
                        "We currently only support unicode strings. Got unsupported kind: {kind}"
                    )
                }

                // Each utf32 char is 4 bytes
                let num_chars_per_item = itemsize / 4;

                let target_shape: Vec<usize> = item.getattr("shape").unwrap().extract().unwrap();

                let item = if target_shape.is_empty() {
                    // Make scalars into 1d arrays
                    item.call_method1("reshape", (1,)).unwrap()
                } else {
                    item
                };

                // View as uint32
                let view = item.call_method1("view", ("uint32",)).unwrap();

                // Convert to np array
                let view: &PyArrayDyn<u32> = view.extract().unwrap();

                // TODO: handle not contiguous
                let data = unsafe { view.as_slice().unwrap() };

                // For each elem
                let data = data
                    .chunks(num_chars_per_item)
                    .map(|item| {
                        let iter = item
                            .iter()
                            // Reverse and remove trailing zeros
                            .rev()
                            .skip_while(|item| **item == 0);

                        // Convert to codepoints
                        let chars: Vec<char> = widestring::decode_utf32(iter.copied())
                            .map(|v| v.unwrap())
                            .collect();

                        // Reverse again and collect into a string
                        chars.into_iter().rev().collect()
                    })
                    .collect();

                let arr = ndarray::ArrayD::from_shape_vec(target_shape, data).unwrap();

                let out = StringPyTensorStorage::new(arr);

                Tensor::String(out)
            }

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
            Tensor::String(item) => {
                // Strings are a bit more complex and require conversion
                // This is unfortunate but necessary because most frameworks have different internal representations of strings
                let view = item.view();

                // First, convert to utf-32
                let strings: Vec<Vec<u32>> = view
                    .iter()
                    .map(|item| widestring::encode_utf32(item.chars()).collect())
                    .collect();

                // Find the length of the longest one
                // TODO: handle empty string arrays
                let longest_string_len_chars = strings.iter().map(|v| v.len()).max().unwrap();

                // Create a u32 array
                let output = numpy::PyArray::<u32, _>::zeros(
                    py,
                    (strings.len() * longest_string_len_chars,),
                    false,
                );
                let data = unsafe { output.as_slice_mut().unwrap() };

                // Copy in the data
                for (idx, item) in strings.into_iter().enumerate() {
                    let offset = idx * longest_string_len_chars;

                    let copy_len = longest_string_len_chars.min(item.len());

                    data[offset..offset + copy_len].copy_from_slice(&item);
                }

                // View it as a unicode array
                let out: &PyAny = output.into();
                let out = out
                    .call_method1("view", (format!("<U{longest_string_len_chars}"),))
                    .unwrap();

                // Reshape it
                let out = out
                    .call_method1("reshape", (PyTuple::new(py, view.shape()),))
                    .unwrap();

                out.into()
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
