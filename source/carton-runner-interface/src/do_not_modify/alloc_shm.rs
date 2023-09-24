use std::{
    collections::BTreeMap,
    marker::PhantomData,
    ops::Bound::Included,
    os::fd::RawFd,
    sync::{atomic::AtomicU64, Arc, Mutex, Weak},
};

use carton_macros::for_each_numeric_carton_type;
use dashmap::DashMap;
use once_cell::sync::Lazy;

use super::{
    alloc::{AsPtr, NumericTensorType, TypedAlloc},
    alloc_pool::{AllocItem, PoolAllocator, PoolItem},
    storage::TensorStorage,
};

#[derive(Debug)]
pub enum SHMTensorStorage {
    Numeric {
        // The region
        region: PoolItem<Arc<SHMRegion>>,

        // The offset into the shared memory region
        offset: usize,
    },

    // Strings are currently stored inline
    String(PoolItem<Vec<String>>),
}

impl<T> AsPtr<T> for SHMTensorStorage {
    /// Get a view of this tensor
    fn as_ptr(&self) -> *const T {
        match self {
            SHMTensorStorage::Numeric { region, offset } => (region.start_addr + *offset) as _,
            // TODO: this should fail if T is not String. Figure out how to do that without specialization
            SHMTensorStorage::String(s) => s.as_ptr() as _,
        }
    }

    /// Get a mut view of this tensor
    fn as_mut_ptr(&mut self) -> *mut T {
        match self {
            SHMTensorStorage::Numeric { region, offset } => (region.start_addr + *offset) as _,
            // TODO: this should fail if T is not String. Figure out how to do that without specialization
            SHMTensorStorage::String(s) => s.as_mut_ptr() as _,
        }
    }
}

enum MemoryMarker {
    ShmRegionStart(SHMRegionID),
    ShmRegionEnd(SHMRegionID),
}

#[derive(Hash, PartialEq, Eq, Debug, Clone, Copy)]
struct SHMRegionID(u64);

// An ID generator for SHMRegionID
static SHM_REGION_ID_GEN: AtomicU64 = AtomicU64::new(0);

// TODO profile and see if we need to refactor to remove this mutex
// Map from address to marker
static ADDR_SPACE: Lazy<Mutex<BTreeMap<usize, MemoryMarker>>> =
    Lazy::new(|| Mutex::new(Default::default()));

// Weak references to all the regions we've allocated
static ALLOCATED_REGIONS: Lazy<DashMap<SHMRegionID, Weak<SHMRegion>>> =
    Lazy::new(|| Default::default());

#[derive(Debug)]
pub struct SHMRegion {
    id: SHMRegionID,
    fd: RawFd,
    start_addr: usize,
    len: usize,
}

/// Use memfd_create to create a new shm region
#[cfg(not(target_os = "macos"))]
unsafe fn memfd_create() -> RawFd {
    libc::memfd_create(b"carton_memfd\0" as *const u8 as _, 0)
}

#[cfg(target_os = "macos")]
unsafe fn memfd_create() -> RawFd {
    static COUNTER: AtomicU64 = AtomicU64::new(0);

    // Generate a path
    let shmpath = format!(
        "/carton_shm_{}_{}\0",
        std::process::id(),
        COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    );

    // memfd_create doesn't exist on mac so we'll use shm_open
    let fd = libc::shm_open(
        shmpath.as_ptr() as _,
        libc::O_CREAT | libc::O_EXCL | libc::O_RDWR,
        (libc::S_IRUSR | libc::S_IWUSR) as libc::c_uint,
    );

    libc::shm_unlink(shmpath.as_ptr() as _);

    fd
}

impl SHMRegion {
    /// Allocate a new shared memory region
    fn new(size_bytes: usize) -> Arc<Self> {
        unsafe {
            let fd = memfd_create();
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
            let region = SHMRegion {
                id: SHMRegionID(
                    SHM_REGION_ID_GEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                ),
                fd,
                start_addr: addr as _,
                len: size_bytes,
            };

            // Mark the beginning and end of the region as allocated
            {
                let mut guard = ADDR_SPACE.lock().unwrap();
                guard.insert(addr as _, MemoryMarker::ShmRegionStart(region.id));
                guard.insert(
                    addr as usize + size_bytes,
                    MemoryMarker::ShmRegionEnd(region.id),
                );
            }

            // Wrap in an arc and insert it into our map of allocated regions
            let id = region.id;
            let out = Arc::new(region);
            ALLOCATED_REGIONS.insert(id, Arc::downgrade(&out));

            out
        }
    }
}

impl Drop for SHMRegion {
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

            // Unmark the beginning and end of the region
            {
                let mut guard = ADDR_SPACE.lock().unwrap();
                guard.remove(&self.start_addr);
                guard.remove(&(self.start_addr + self.len));
            }

            // Remove the allocated region
            ALLOCATED_REGIONS.remove(&self.id);
        }
    }
}

impl AllocItem for Arc<SHMRegion> {
    fn new(numel: usize) -> Self {
        SHMRegion::new(numel)
    }

    fn len(&self) -> usize {
        self.len
    }
}

pub struct SHMAllocator {
    use_pool: bool,
    numeric: Arc<PoolAllocator<Arc<SHMRegion>>>,
    string: Arc<PoolAllocator<Vec<String>>>,
}

impl SHMAllocator {
    pub(crate) fn new() -> Self {
        Self {
            use_pool: true,
            numeric: Arc::new(PoolAllocator::new()),
            string: Arc::new(PoolAllocator::new()),
        }
    }

    #[cfg(feature = "benchmark")]
    pub(crate) fn without_pool() -> Self {
        Self {
            use_pool: false,
            numeric: Arc::new(PoolAllocator::new()),
            string: Arc::new(PoolAllocator::new()),
        }
    }

    /// Get the SHM region containing addr (if any)
    fn get_shm_region(addr: usize) -> Option<Arc<SHMRegion>> {
        match ADDR_SPACE
            .lock()
            .unwrap()
            .range((Included(&0), Included(&addr)))
            .next_back()
            .map(|(_, v)| v)
        {
            // The highest marker <= the target address is the start of a shm region (meaning addr is in an shm region)
            Some(MemoryMarker::ShmRegionStart(id)) => match ALLOCATED_REGIONS.get(id) {
                Some(region) => region.upgrade(),
                None => None,
            },
            Some(MemoryMarker::ShmRegionEnd(_)) => None,
            None => None,
        }
    }
}

for_each_numeric_carton_type! {
    $(
        /// We're using a macro here instead of a generic impl because rust gives misleading error messages otherwise.
        impl TypedAlloc<$RustType> for SHMAllocator {
            type Output = SHMTensorStorage;

            fn alloc(&self, numel: usize) -> Self::Output {
                // We need to convert to size_bytes
                let size_bytes = numel * std::mem::size_of::<$RustType>();
                let out = if !self.use_pool {
                    SHMRegion::new(size_bytes).into()
                } else {
                    self.numeric.alloc(size_bytes)
                };

                SHMTensorStorage::Numeric {region: out, offset: 0 }
            }
        }
    )*
}

impl TypedAlloc<String> for SHMAllocator {
    type Output = SHMTensorStorage;

    fn alloc(&self, numel: usize) -> Self::Output {
        let out = if !self.use_pool {
            vec![String::default(); numel].into()
        } else {
            self.string.alloc(numel)
        };

        SHMTensorStorage::String(out)
    }
}

/// "Conversion" of an ArrayViewD to SHM
///
/// There are two options:
/// 1. We copy the data
/// 2. The data backing the view is already in shared memory
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
impl<T: NumericTensorType> From<ndarray::ArrayViewD<'_, T>> for TensorStorage<T, SHMTensorStorage>
where
    SHMAllocator: TypedAlloc<T, Output = SHMTensorStorage>,
{
    fn from(view: ndarray::ArrayViewD<'_, T>) -> Self {
        let ptr = view.as_ptr();

        // If ptr is within a shared memory range we've previously allocated, create a new ndarray with different
        // storage, but the same pointer.
        match SHMAllocator::get_shm_region(ptr as usize) {
            Some(region) => TensorStorage {
                data: SHMTensorStorage::Numeric {
                    offset: ptr as usize - region.start_addr,
                    region: region.into(),
                },
                shape: view.shape().into_iter().map(|v| *v as _).collect(),
                strides: Some(
                    view.strides()
                        .into_iter()
                        .map(|v| (*v).try_into().unwrap())
                        .collect(),
                ),
                pd: PhantomData,
            },
            None => {
                // TODO WARN

                // We need to make a copy
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
    }
}

/// Just need to copy for strings
impl From<ndarray::ArrayViewD<'_, String>> for TensorStorage<String, SHMTensorStorage> {
    fn from(view: ndarray::ArrayViewD<'_, String>) -> Self {
        // Alloc a tensor
        let mut out = alloc_tensor(view.shape().iter().map(|v| (*v) as _).collect());

        // Can't memcpy
        out.view_mut().assign(&view);

        out
    }
}

// Allocates a contiguous tensor with a shape and type
#[cfg(feature = "benchmark")]
pub fn alloc_tensor_no_pool<T: Default + Clone>(
    shape: Vec<u64>,
) -> TensorStorage<T, SHMTensorStorage>
where
    SHMAllocator: TypedAlloc<T, Output = SHMTensorStorage>,
{
    static POOL_ALLOCATOR: Lazy<SHMAllocator> = Lazy::new(|| SHMAllocator::without_pool());

    let numel = shape.iter().product::<u64>().max(1) as usize;

    let data = <SHMAllocator as TypedAlloc<T>>::alloc(&POOL_ALLOCATOR, numel);

    TensorStorage {
        data,
        shape,
        strides: None,
        pd: PhantomData,
    }
}

pub fn alloc_tensor<T: Default + Clone>(shape: Vec<u64>) -> TensorStorage<T, SHMTensorStorage>
where
    SHMAllocator: TypedAlloc<T, Output = SHMTensorStorage>,
{
    static POOL_ALLOCATOR: Lazy<SHMAllocator> = Lazy::new(|| SHMAllocator::new());

    let numel = shape.iter().product::<u64>().max(1) as usize;

    let data = <SHMAllocator as TypedAlloc<T>>::alloc(&POOL_ALLOCATOR, numel);

    TensorStorage {
        data,
        shape,
        strides: None,
        pd: PhantomData,
    }
}
