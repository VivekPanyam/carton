use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
    sync::Arc,
};

use carton_macros::for_each_numeric_carton_type;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};

use super::{
    alloc::{AsPtr, NumericTensorType, TypedAlloc},
    alloc_pool::{PoolAllocator, PoolItem},
    storage::TensorStorage,
};

pub struct InlineAllocator {
    use_pool: bool,
    numeric: Arc<PoolAllocator<Vec<u8>>>,
    string: Arc<PoolAllocator<Vec<String>>>,
}

impl InlineAllocator {
    pub(crate) fn new() -> Self {
        Self {
            use_pool: true,
            numeric: Arc::new(PoolAllocator::new()),
            string: Arc::new(PoolAllocator::new()),
        }
    }

    pub(crate) fn without_pool() -> Self {
        Self {
            use_pool: false,
            numeric: Arc::new(PoolAllocator::new()),
            string: Arc::new(PoolAllocator::new()),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub enum InlineTensorStorage {
    Numeric(#[serde(with = "serde_bytes")] PoolItem<Vec<u8>>),
    String(PoolItem<Vec<String>>),
}

impl<T> AsPtr<T> for InlineTensorStorage {
    /// Get a view of this tensor
    fn as_ptr(&self) -> *const T {
        match self {
            InlineTensorStorage::Numeric(s) => s.deref().as_ptr() as _,
            // TODO: this should fail if T is not String. Figure out how to do that without specialization
            InlineTensorStorage::String(s) => s.as_ptr() as _,
        }
    }

    /// Get a mut view of this tensor
    fn as_mut_ptr(&mut self) -> *mut T {
        match self {
            InlineTensorStorage::Numeric(s) => s.deref_mut().as_mut_ptr() as _,
            // TODO: this should fail if T is not String. Figure out how to do that without specialization
            InlineTensorStorage::String(s) => s.as_mut_ptr() as _,
        }
    }
}

for_each_numeric_carton_type! {
    $(
        /// We're using a macro here instead of a generic impl because rust gives misleading error messages otherwise.
        impl TypedAlloc<$RustType> for InlineAllocator {
            type Output = InlineTensorStorage;

            fn alloc(&self, numel: usize) -> Self::Output {
                // We need to convert to size_bytes since we always use a Vec<u8>
                let size_bytes = numel * std::mem::size_of::<$RustType>();
                let out = if !self.use_pool {
                    vec![0u8; size_bytes].into()
                } else {
                    self.numeric.alloc(size_bytes)
                };

                InlineTensorStorage::Numeric(out)
            }
        }
    )*
}

impl TypedAlloc<String> for InlineAllocator {
    type Output = InlineTensorStorage;

    fn alloc(&self, numel: usize) -> Self::Output {
        let out = if !self.use_pool {
            vec![String::default(); numel].into()
        } else {
            self.string.alloc(numel)
        };

        InlineTensorStorage::String(out)
    }
}

// Copy the data
impl<T: NumericTensorType + Default + Copy> From<ndarray::ArrayViewD<'_, T>>
    for TensorStorage<T, InlineTensorStorage>
where
    InlineAllocator: TypedAlloc<T, Output = InlineTensorStorage>,
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
    InlineAllocator: TypedAlloc<T, Output = InlineTensorStorage>,
{
    static POOL_ALLOCATOR: Lazy<InlineAllocator> = Lazy::new(|| InlineAllocator::without_pool());

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
    InlineAllocator: TypedAlloc<T, Output = InlineTensorStorage>,
{
    static POOL_ALLOCATOR: Lazy<InlineAllocator> = Lazy::new(|| InlineAllocator::new());

    let numel = shape.iter().product::<u64>() as usize;

    let data = POOL_ALLOCATOR.alloc(numel);

    TensorStorage {
        data,
        shape,
        strides: None,
        pd: PhantomData,
    }
}
