//! TensorStorage that is stored inline

use carton_macros::for_each_numeric_carton_type;
use ndarray::{ShapeBuilder, StrideShape};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct TensorStorage<T> {
    // TODO: use a vec<u8> for primitive types
    data: Vec<T>,
    shape: Vec<u64>,
    strides: Option<Vec<u64>>,
}

impl<T> TensorStorage<T> {
    fn get_shape(&self) -> StrideShape<ndarray::IxDyn> {
        match &self.strides {
            None => self
                .shape
                .iter()
                .map(|v| *v as usize)
                .collect::<Vec<_>>()
                .into(),
            Some(strides) => self
                .shape
                .iter()
                .map(|v| *v as usize)
                .collect::<Vec<_>>()
                .strides(strides.iter().map(|v| (*v).try_into().unwrap()).collect())
                .into(),
        }
    }

    pub fn view(&self) -> ndarray::ArrayViewD<T> {
        let data = self.data.as_ptr();
        unsafe { ndarray::ArrayView::from_shape_ptr(self.get_shape(), data) }
    }

    pub fn view_mut(&mut self) -> ndarray::ArrayViewMutD<T> {
        let data = self.data.as_mut_ptr();
        unsafe { ndarray::ArrayViewMut::from_shape_ptr(self.get_shape(), data) }
    }
}

// Copy the data
impl<T: NumericCartonType> From<ndarray::ArrayViewD<'_, T>> for TensorStorage<T> {
    fn from(view: ndarray::ArrayViewD<'_, T>) -> Self {
        // Alloc a tensor
        let mut out = alloc_tensor(view.shape().iter().map(|v| (*v) as _).collect());

        if view.is_standard_layout() {
            // We can just memcpy the data
            out.view_mut()
                .as_slice_mut()
                .unwrap()
                .copy_from_slice(view.as_slice().unwrap())
        } else {
            out.view_mut().assign(&view);
        }

        out
    }
}

impl From<ndarray::ArrayViewD<'_, String>> for TensorStorage<String> {
    fn from(view: ndarray::ArrayViewD<'_, String>) -> Self {
        // Alloc a tensor
        let mut out = alloc_tensor(view.shape().iter().map(|v| (*v) as _).collect());

        // Can't memcpy
        out.view_mut().assign(&view);

        out
    }
}

// Allocates a contiguous tensor with a shape and type
pub(crate) fn alloc_tensor<T: Default + Clone>(shape: Vec<u64>) -> TensorStorage<T> {
    let numel: u64 = shape.iter().product();
    let data: Vec<T> = vec![T::default(); numel as _];

    TensorStorage {
        data,
        shape,
        strides: None,
    }
}

trait NumericCartonType: Default + Copy {}

for_each_numeric_carton_type! {
    $(
        impl NumericCartonType for $RustType {}
    )*
}
