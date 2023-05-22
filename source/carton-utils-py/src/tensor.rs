use numpy::PyArrayDyn;
use pyo3::{types::PyTuple, AsPyPointer, FromPyObject, PyAny, PyDowncastError, PyObject, Python};

pub struct PyStringArrayType<'a> {
    inner: &'a PyAny,
}

impl<'py> FromPyObject<'py> for PyStringArrayType<'py> {
    fn extract(ob: &'py PyAny) -> pyo3::PyResult<Self> {
        unsafe {
            if numpy::npyffi::PyArray_Check(ob.py(), ob.as_ptr()) == 0 {
                return Err(PyDowncastError::new(ob, "PyStringArray").into());
            }
        }

        let kind: char = ob
            .getattr("dtype")
            .unwrap()
            .getattr("kind")
            .unwrap()
            .extract()
            .unwrap();

        if kind != 'U' {
            // Only unicode strings are supported
            return Err(PyDowncastError::new(ob, "PyStringArray").into());
        }

        Ok(PyStringArrayType { inner: ob })
    }
}

impl<'a> PyStringArrayType<'a> {
    /// Strings are a bit complex and require conversion
    /// This is unfortunate but necessary because most frameworks have different internal representations of strings
    pub fn to_ndarray(&self) -> ndarray::ArrayD<String> {
        let item = self.inner;
        let itemsize: usize = item.getattr("itemsize").unwrap().extract().unwrap();

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

        ndarray::ArrayD::from_shape_vec(target_shape, data).unwrap()
    }

    /// Strings are a bit complex and require conversion
    /// This is unfortunate but necessary because most frameworks have different internal representations of strings
    pub fn from_ndarray(py: Python, view: ndarray::ArrayViewD<String>) -> PyObject {
        // First, convert to utf-32
        let strings: Vec<Vec<u32>> = view
            .iter()
            .map(|item| widestring::encode_utf32(item.chars()).collect())
            .collect();

        // Find the length of the longest one
        // TODO: handle empty string arrays
        let longest_string_len_chars = strings.iter().map(|v| v.len()).max().unwrap();

        // Create a u32 array
        let output =
            numpy::PyArray::<u32, _>::zeros(py, (strings.len() * longest_string_len_chars,), false);
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
}
