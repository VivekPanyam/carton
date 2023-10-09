// Copyright 2023 Vivek Panyam
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::ffi::c_void;

use crate::{carton::Carton, tensormap::CartonTensorMap};

/// The type of a callback to async Carton functions.
/// These callbacks run on internal threads managed by Carton.
///
/// IMPORTANT: these callbacks should not block or do CPU-intensive work.
/// Doing so could block Carton's internal event system.
/// If you would like callbacks on a thread without these restrictions, see `CartonAsyncNotifier`
pub type CartonLoadCallback =
    extern "C" fn(result: *mut Carton, status: CartonStatus, arg: *mut c_void);
pub type CartonInferCallback =
    extern "C" fn(result: *mut CartonTensorMap, status: CartonStatus, arg: *mut c_void);
// pub type CartonTensorCallback =
//     extern "C" fn(result: *mut CartonTensor, status: CartonStatus, arg: *mut c_void);
pub type CartonNotifierCallback =
    extern "C" fn(result: *mut c_void, status: CartonStatus, arg: *mut c_void);

// Unforunately, we have to manually spell this out because cbindgen macro expansion requires using the nightly compiler
/// cbindgen:rename-all=QualifiedScreamingSnakeCase
#[repr(C)]
pub enum DataType {
    Float,
    Double,
    String,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

/// cbindgen:rename-all=QualifiedScreamingSnakeCase
#[derive(Debug)]
#[repr(C)]
pub enum CartonStatus {
    /// The operation was completed successfully
    Success,

    /// There were no async tasks ready
    NoAsyncTasksReady,
}

#[derive(Debug)]
#[repr(transparent)]
pub struct CallbackArg {
    pub(crate) inner: *mut c_void,
}

impl From<CallbackArg> for *mut c_void {
    fn from(value: CallbackArg) -> Self {
        value.inner
    }
}

unsafe impl Send for CallbackArg {}
unsafe impl Sync for CallbackArg {}
