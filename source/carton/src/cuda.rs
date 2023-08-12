//! Minimal bindings around CUDA that let us fetch devices
#![allow(non_snake_case)]
use dlopen::wrapper::{Container, WrapperApi};
use dlopen_derive::WrapperApi;
use lazy_static::{__Deref, lazy_static};
use uuid::Uuid;

#[derive(WrapperApi)]
struct Cuda {
    cuInit: unsafe extern "C" fn(flags: u32) -> u32,
    cuDeviceGet: unsafe extern "C" fn(device: *mut i32, idx: i32) -> u32,
    cuDeviceGetUuid_v2: unsafe extern "C" fn(uuid: *mut [u8; 16], device: i32) -> u32,
}

enum CudaState {
    Loaded(Container<Cuda>),
    LoadFailed,
    InitFailed,
}

lazy_static! {
    static ref CUDA: CudaState = unsafe {
        if let Ok(cuda) = Container::<Cuda>::load("libcuda.so.1") {
            if cuda.cuInit(0) != 0 {
                CudaState::InitFailed
            } else {
                CudaState::Loaded(cuda)
            }
        } else {
            CudaState::LoadFailed
        }
    };
}

pub(crate) fn get_uuid_for_device(ordinal: u32) -> Option<String> {
    match CUDA.deref() {
        CudaState::Loaded(cuda) => {
            unsafe {
                let mut device = 0;
                if cuda.cuDeviceGet(&mut device as _, ordinal as _) != 0 {
                    // Invalid device
                    log::warn!("Tried to get index of cuda device that doesn't exist.");
                    return None;
                }

                let mut uuid = [0; 16];
                assert_eq!(cuda.cuDeviceGetUuid_v2(&mut uuid as _, device), 0);

                // TODO: do we need to do something else for MIG?
                Some(
                    "GPU-".to_string() + &Uuid::from_slice(&uuid).unwrap().hyphenated().to_string(),
                )
            }
        }
        CudaState::LoadFailed => {
            log::warn!("Tried to get a CUDA device but failed to load libcuda.so.1");
            None
        }
        CudaState::InitFailed => {
            log::trace!(
                "Loaded libcuda but cuInit failed. This can happen if no GPUs are visible."
            );
            None
        }
    }
}

mod tests {
    #[test]
    fn basic_test() {
        let uuid = super::get_uuid_for_device(1);
        println!("{uuid:#?}");
    }
}
