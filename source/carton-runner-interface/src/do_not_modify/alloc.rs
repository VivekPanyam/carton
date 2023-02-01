use std::{
    fmt::Debug,
    ops::{Deref, DerefMut},
    sync::Arc,
};

use carton_macros::for_each_numeric_carton_type;
use serde::{Deserialize, Serialize};

use super::alloc_pool::{AllocItem, PoolAllocator, PoolItem};

/// Numeric tensor types supported by this version of the runner interface
pub(crate) trait NumericTensorType: Default + Copy {}

for_each_numeric_carton_type! {
    $(
        impl NumericTensorType for $RustType {}
    )*
}

impl<T: Default + Clone> AllocItem for Vec<T> {
    fn new(numel: usize) -> Self {
        vec![T::default(); numel]
    }

    fn len(&self) -> usize {
        self.len()
    }
}

pub trait AsPtr<T> {
    /// Get a view of this tensor
    fn as_ptr(&self) -> *const T;

    /// Get a mut view of this tensor
    fn as_mut_ptr(&mut self) -> *mut T;
}

pub trait TypedAlloc<T> {
    type Output: AsPtr<T>;

    fn alloc(&self, numel: usize) -> Self::Output;
}

pub struct Allocator {
    use_pool: bool,
    numeric: Arc<PoolAllocator<Vec<u8>>>,
    string: Arc<PoolAllocator<Vec<String>>>,
}

impl Allocator {
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
        impl TypedAlloc<$RustType> for Allocator {
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

impl TypedAlloc<String> for Allocator {
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
