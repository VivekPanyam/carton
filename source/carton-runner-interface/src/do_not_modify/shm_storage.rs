//! Allocate SHM regions
use dashmap::DashMap;
use lazy_static::lazy_static;
use std::collections::BTreeMap;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::os::fd::RawFd;
use std::sync::atomic::AtomicU64;
use std::sync::{Mutex, Weak};
use std::{ops::Bound::Included, sync::Arc};

use crate::do_not_modify::ndarray::{NDarray, Storage};

enum MemoryMarker {
    ShmRegionStart(SHMRegionID),
    ShmRegionEnd(SHMRegionID),
}

#[derive(Hash, PartialEq, Eq, Debug, Clone, Copy)]
struct SHMRegionID(u64);

#[derive(Debug)]
struct SHMRegionInner {
    id: SHMRegionID,
    fd: RawFd,
    start_addr: usize,
    len: usize,
}

impl Drop for SHMRegionInner {
    fn drop(&mut self) {
        unsafe {
            // Unmap
            let res = libc::munmap(self.start_addr as _, self.len);
            if res != 0 {
                panic!("munmap failed")
            }

            // Close the fd
            let res = libc::close(self.fd);
            if res != 0 {
                panic!("close failed")
            }
        }
    }
}

#[derive(Debug)]
struct SHMRegion(Option<SHMRegionInner>);

impl SHMRegion {
    fn new(inner: SHMRegionInner) -> Self {
        Self(Some(inner))
    }
}

impl Drop for SHMRegion {
    fn drop(&mut self) {
        SHM_ALLOCATOR.return_for_reuse(self.0.take().unwrap())
    }
}

pub(crate) struct SHMAllocator {
    shm_region_id_gen: AtomicU64,

    // TODO profile and see if we need to refactor to remove this mutex
    // Map from address to marker
    addr_space: Mutex<BTreeMap<usize, MemoryMarker>>,

    /// Weak references to all the regions we've allocated
    allocated_regions: DashMap<SHMRegionID, Weak<SHMRegion>>,

    // TODO: handle cache eviction every once in a while
    reusable_regions: DashMap<usize, Vec<SHMRegionInner>>,
}

impl SHMAllocator {
    fn new() -> Self {
        Self {
            addr_space: Default::default(),
            allocated_regions: Default::default(),
            shm_region_id_gen: 0.into(),
            reusable_regions: Default::default(),
        }
    }

    /// Get the SHM region containing addr (if any)
    fn get_shm_region(&self, addr: usize) -> Option<Arc<SHMRegion>> {
        match self
            .addr_space
            .lock()
            .unwrap()
            .range((Included(&0), Included(&addr)))
            .next_back()
            .map(|(_, v)| v)
        {
            // The highest marker <= the target address is the start of a shm region (meaning addr is in an shm region)
            Some(MemoryMarker::ShmRegionStart(id)) => match self.allocated_regions.get(id) {
                Some(region) => region.upgrade(),
                None => None,
            },
            Some(MemoryMarker::ShmRegionEnd(_)) => None,
            None => None,
        }
    }

    fn alloc(&self, size_bytes: usize) -> Arc<SHMRegion> {
        // Check if we can reuse a region
        if let Some(mut reusable) = self.reusable_regions.get_mut(&size_bytes) {
            // Pop the last element. This makes an lru strategy work better because the front of the vec is
            // not touched unless it needs to be
            if let Some(item) = reusable.pop() {
                let id = item.id;
                let out = Arc::new(SHMRegion::new(item));
                self.allocated_regions.insert(id, Arc::downgrade(&out));

                return out;
            }
        }

        // We need to allocate
        unsafe {
            // Use memfd_create to create a new shm region
            let fd = libc::memfd_create("carton_memfd".as_ptr() as _, 0);
            if fd == -1 {
                panic!("memfd_create failed")
            }

            // Set the size
            if libc::ftruncate(fd, size_bytes as _) == -1 {
                panic!("ftruncate failed")
            }

            // MMAP
            let addr = libc::mmap(
                std::ptr::null_mut(),
                size_bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            );
            if addr == libc::MAP_FAILED {
                panic!("mmap failed");
            }

            // Create an inner region
            let region = SHMRegionInner {
                id: SHMRegionID(
                    self.shm_region_id_gen
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                ),
                fd,
                start_addr: addr as _,
                len: size_bytes,
            };

            // Mark the beginning and end of the region as allocated
            {
                let mut guard = self.addr_space.lock().unwrap();
                guard.insert(addr as _, MemoryMarker::ShmRegionStart(region.id));
                guard.insert(
                    addr as usize + size_bytes,
                    MemoryMarker::ShmRegionEnd(region.id),
                );
            }

            // Wrap it and return
            let id = region.id;
            let out = Arc::new(SHMRegion::new(region));
            self.allocated_regions.insert(id, Arc::downgrade(&out));

            return out;
        }
    }

    /// Returns a region to the pool
    fn return_for_reuse(&self, region: SHMRegionInner) {
        self.allocated_regions.remove(&region.id);
        self.reusable_regions
            .entry(region.len)
            .or_default()
            .push(region)
    }
}

lazy_static! {
    static ref SHM_ALLOCATOR: SHMAllocator = SHMAllocator::new();
}

#[derive(Debug)]
pub struct SHMStorage<T> {
    region: Arc<SHMRegion>,
    _pd: PhantomData<T>,
}

impl<T> Storage<T> for SHMStorage<T>
where
    T: Debug,
{
    fn get(&self) -> &[T] {
        let region = self.region.0.as_ref().unwrap();
        unsafe { std::slice::from_raw_parts(region.start_addr as _, region.len as _) }
    }
}

/// "Conversion" of an NDarray to SHM
///
/// There are two options:
/// 1. We copy the data
/// 2. The data backing the NDarray is already in shared memory
///
/// The second option will only happen if the data is in shared memory that we allocated.
///
/// Therefore, we implement this conversion as follows
///
/// 1. If the data pointer of the tensor is within a shared memory region we allocated,
///    get that shm region and create a new tensor with the same shape, strides,
///    and data pointer.
///
/// 2. Otherwise, make a complete copy of the tensor
///
///
/// `[bindings] -> [core library] -> [runner] -> [framework]`
///
/// can use shared memory the whole way if user code uses `alloc_tensor` to allocate tensors with Carton.
///
/// `[framework] -> [runner] -> [core library] -> [bindings]`
///
/// unfortunately requires a copy in many cases as we can't easily control the allocator used by the
/// underlying ML frameworks.
///
/// This could something interesting to explore as an optimization if necessary.
///
impl<T, S> From<NDarray<T, S>> for NDarray<T, SHMStorage<T>>
where
    T: Debug + Clone + Copy,
    S: AsRef<[T]> + Debug,
{
    fn from(value: NDarray<T, S>) -> Self {
        let ptr = value.data().as_ptr();

        // If ptr is within a shared memory range we've previously allocated, create a new ndarray with different
        // storage, but the same pointer.
        match SHM_ALLOCATOR.get_shm_region(ptr as usize) {
            Some(region) => {
                let storage = SHMStorage {
                    region,
                    _pd: PhantomData,
                };

                NDarray::from_shape_strides_storage(
                    value.shape().clone(),
                    value.strides().clone(),
                    storage,
                )
            }
            None => {
                // TODO WARN

                // We need to make a copy
                let view = value.view();

                // TODO: THIS IS NOT CORRECT FOR STRINGS
                let out = alloc_tensor(view.shape().iter().map(|v| (*v) as _).collect());

                if view.is_standard_layout() {
                    // We can just memcpy the data
                    out.data_mut().copy_from_slice(value.data())
                } else {
                    out.view_mut().assign(&value.view());
                }

                out
            }
        }
    }
}

// TODO: THIS IS NOT CORRECT FOR STRINGS
// Allocates a contiguous tensor with a shape and type
pub(crate) fn alloc_tensor<T: Debug>(shape: Vec<u64>) -> NDarray<T, SHMStorage<T>> {
    // Multiply all the elements of shape
    let numel = shape.iter().product::<u64>() as usize;

    let bytes_per_elem = std::mem::size_of::<T>();
    let contiguous_size_bytes = numel * bytes_per_elem;

    let region = SHM_ALLOCATOR.alloc(contiguous_size_bytes);
    let storage = SHMStorage {
        region,
        _pd: PhantomData,
    };

    NDarray::from_shape_storage(shape, storage)
}

pub use SHMStorage as TensorStorage;
