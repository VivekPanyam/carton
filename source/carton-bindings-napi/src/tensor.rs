use std::marker::PhantomData;

use carton_core::types::{for_each_carton_type, TypedStorage};
use napi::bindgen_prelude::Buffer;
use ndarray::ShapeBuilder;

#[napi(object)]
#[derive(Clone)]
pub struct Tensor {
    pub buffer: Buffer,
    pub shape: Vec<u32>,
    pub dtype: String,
    pub stride: Vec<u32>,
}

// TODO: describe why this is safe
unsafe impl Send for Tensor {}
unsafe impl Sync for Tensor {}

pub(crate) struct TypedNodeTensorStorage<T> {
    inner: Tensor,
    _phantom: PhantomData<T>,
}

impl<T> TypedStorage<T> for TypedNodeTensorStorage<T> {
    fn view(&self) -> ndarray::ArrayViewD<T> {
        unsafe {
            ndarray::ArrayView::from_shape_ptr(
                self.inner
                    .shape
                    .iter()
                    .map(|v| *v as usize)
                    .collect::<Vec<_>>()
                    .strides(self.inner.stride.iter().map(|v| *v as usize).collect()),
                self.inner.buffer.as_ptr() as *const T,
            )
        }
    }

    fn view_mut(&mut self) -> ndarray::ArrayViewMutD<T> {
        unsafe {
            ndarray::ArrayViewMut::from_shape_ptr(
                self.inner
                    .shape
                    .iter()
                    .map(|v| *v as usize)
                    .collect::<Vec<_>>()
                    .strides(self.inner.stride.iter().map(|v| *v as usize).collect()),
                self.inner.buffer.as_mut_ptr() as *mut T,
            )
        }
    }
}

impl From<Tensor> for carton_core::types::Tensor {
    fn from(value: Tensor) -> Self {
        for_each_carton_type! {
            return match value.dtype.as_str() {
                $(
                    $TypeStr => carton_core::types::Tensor::$CartonType(TypedNodeTensorStorage { inner: value, _phantom: PhantomData}.into()),
                )*
                dtype => panic!("Got unknown dtype: {dtype}"),
            }
        }
    }
}

impl From<carton_core::types::Tensor> for Tensor {
    fn from(value: carton_core::types::Tensor) -> Self {
        (&value).into()
    }
}

// TODO: using a &Tensor<T> for now because we're making a copy
impl From<&carton_core::types::Tensor> for Tensor {
    fn from(value: &carton_core::types::Tensor) -> Self {
        for_each_carton_type! {
            return match value {
                $(
                    carton_core::types::Tensor::$CartonType(t) => {
                        // Get the data as a slice
                        // TODO: this can make a copy
                        let view = t.view();
                        let mut standard = view.as_standard_layout();

                        let data = standard.as_slice_mut().unwrap();

                        // View it as a u8 slice
                        let data = unsafe {
                            std::slice::from_raw_parts_mut(
                                data.as_mut_ptr() as *mut u8,
                                data.len() * std::mem::size_of::<$RustType>(),
                            )
                        };

                        // TODO: this makes a copy
                        Tensor {
                            buffer: data.to_owned().into(),
                            shape: view.shape().iter().map(|v| *v as _).collect(),
                            dtype: $TypeStr.to_owned(),
                            stride: view.strides().iter().map(|v| *v as _).collect(),
                        }
                    },
                )*
                carton_core::types::Tensor::NestedTensor(_) => panic!("Nested tensor output not implemented yet"),
            }

        }
    }
}
