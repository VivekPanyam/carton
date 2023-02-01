//! A reuse pool for allocators

use std::{
    ops::{Deref, DerefMut},
    sync::{Arc, Weak},
};

use dashmap::DashMap;
use serde::{Deserialize, Serialize};

/// The item wrapper handed out by the pool
/// IMPORTANT: changing this type can affect the wire protocol. Be careful
#[derive(Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PoolItem<T: AllocItem> {
    #[serde(skip)]
    allocator: Option<Weak<PoolAllocator<T>>>,

    inner: Option<T>,
}

impl serde_bytes::Serialize for PoolItem<Vec<u8>> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serde_bytes::Serialize::serialize(&self.inner, serializer)
    }
}

impl<'de> serde_bytes::Deserialize<'de> for PoolItem<Vec<u8>> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Ok(Self {
            allocator: None,
            inner: serde_bytes::Deserialize::deserialize(deserializer)?,
        })
    }
}

/// Convenience method to convert into a PoolItem with no pool
/// Useful in tests and benchmarks
impl<T: AllocItem> From<T> for PoolItem<T> {
    fn from(value: T) -> Self {
        Self {
            allocator: None,
            inner: Some(value),
        }
    }
}

/// Return the inner item to the pool if we have one and the pool still exists
impl<T: AllocItem> Drop for PoolItem<T> {
    fn drop(&mut self) {
        self.allocator.as_ref().map(|weak| {
            weak.upgrade()
                .map(|alloc| alloc.return_for_reuse(self.inner.take().unwrap()))
        });
    }
}

impl<T: AllocItem> Deref for PoolItem<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.inner.as_ref().unwrap()
    }
}

impl<T: AllocItem> DerefMut for PoolItem<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.inner.as_mut().unwrap()
    }
}

/// An item that can be allocated
pub trait AllocItem {
    fn new(numel: usize) -> Self;

    /// This MUST return the same value as `numel` passed into `new` above
    fn len(&self) -> usize;
}

impl<T: Default + Clone> AllocItem for Vec<T> {
    fn new(numel: usize) -> Self {
        vec![T::default(); numel]
    }

    fn len(&self) -> usize {
        self.len()
    }
}

/// Allocates `T: AllocItem` and attempts to reuse previously allocated and dropped items.
#[derive(Debug)]
pub(crate) struct PoolAllocator<T> {
    // TODO: handle cache eviction every once in a while
    /// A map of items that can be reused
    reusable: DashMap<usize, Vec<T>>,
}

impl<T: AllocItem> PoolAllocator<T> {
    pub(crate) fn new() -> Self {
        Self {
            reusable: Default::default(),
        }
    }

    /// Allocate a `T`. Tries to reuse previous allocations if possible.
    pub(crate) fn alloc(self: &Arc<Self>, numel: usize) -> PoolItem<T> {
        // Check if we can reuse something that was previously allocated
        if let Some(mut reusable) = self.reusable.get_mut(&numel) {
            // Pop the last element. This makes an lru strategy work better because the front of the vec is
            // not touched unless it needs to be
            if let Some(item) = reusable.pop() {
                return PoolItem {
                    allocator: Some(Arc::downgrade(self)),
                    inner: Some(item),
                };
            }
        }

        // We need to allocate
        let item = T::new(numel);
        return PoolItem {
            allocator: Some(Arc::downgrade(self)),
            inner: Some(item),
        };
    }

    fn return_for_reuse(&self, item: T) {
        self.reusable.entry(item.len()).or_default().push(item)
    }
}
