//! TensorStorage that is stored inline

use std::{fmt::Debug, marker::PhantomData};

use ndarray::{ShapeBuilder, StrideShape};
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};

use super::alloc::{Allocator, AsPtr, InlineTensorStorage, NumericTensorType, TypedAlloc};

#[derive(Debug, Serialize, Deserialize)]
pub struct TensorStorage<T, Storage> {
    data: Storage,
    shape: Vec<u64>,
    strides: Option<Vec<u64>>,
    pd: PhantomData<T>,
}

impl<T, Storage> TensorStorage<T, Storage>
where
    Storage: AsPtr<T>,
{
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
impl<T: NumericTensorType + Default + Copy> From<ndarray::ArrayViewD<'_, T>>
    for TensorStorage<T, InlineTensorStorage>
where
    Allocator: TypedAlloc<T, Output = InlineTensorStorage>,
{
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

impl From<ndarray::ArrayViewD<'_, String>> for TensorStorage<String, InlineTensorStorage> {
    fn from(view: ndarray::ArrayViewD<'_, String>) -> Self {
        // Alloc a tensor
        let mut out = alloc_tensor(view.shape().iter().map(|v| (*v) as _).collect());

        // Can't memcpy
        out.view_mut().assign(&view);

        out
    }
}

// Allocates a contiguous tensor with a shape and type
pub fn alloc_tensor_no_pool<T: Default + Clone>(
    shape: Vec<u64>,
) -> TensorStorage<T, InlineTensorStorage>
where
    Allocator: TypedAlloc<T, Output = InlineTensorStorage>,
{
    static POOL_ALLOCATOR: Lazy<Allocator> = Lazy::new(|| Allocator::without_pool());

    let numel = shape.iter().product::<u64>() as usize;

    let data = POOL_ALLOCATOR.alloc(numel);

    TensorStorage {
        data,
        shape,
        strides: None,
        pd: PhantomData,
    }
}

pub fn alloc_tensor<T: Default + Clone>(shape: Vec<u64>) -> TensorStorage<T, InlineTensorStorage>
where
    Allocator: TypedAlloc<T, Output = InlineTensorStorage>,
{
    static POOL_ALLOCATOR: Lazy<Allocator> = Lazy::new(|| Allocator::new());

    let numel = shape.iter().product::<u64>() as usize;

    let data = POOL_ALLOCATOR.alloc(numel);

    TensorStorage {
        data,
        shape,
        strides: None,
        pd: PhantomData,
    }
}
