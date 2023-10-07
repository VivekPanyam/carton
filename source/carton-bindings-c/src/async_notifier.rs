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

use std::{ffi::c_void, sync::atomic::AtomicU64};

use dashmap::DashMap;
use tokio::sync::mpsc;

use crate::types::{CallbackArg, CartonNotifierCallback, CartonStatus};

/// A way to get results of async functions in a less restricted environment than using
/// a callback directly.
///
/// Example usage (error handling omitted for brevity):
/// ```c
/// // Create an async notifier
/// CartonAsyncNotifier* notifier;
/// carton_async_notifier_create(&notifier);
///
/// // ... later ...
///
/// // Define a callback and a callback arg
/// CartonNotifierCallback callback;
/// void *callback_arg = ...; // Anything you want
///
/// // Register the callback
/// carton_async_notifier_register(notifier, &callback, &callback_arg);
///
/// // Run an async function
/// carton_load("/some/path", (CartonLoadCallback)callback, callback_arg);
///
/// // ... anywhere (on the current thread or some other one) ...
/// void *result;
/// void *callback_arg;
/// carton_async_notifier_wait(notifier, &result, &callback_arg);
/// ```
struct CartonAsyncNotifier {
    recv: mpsc::UnboundedReceiver<ResultTransportWrapper>,
    id: u64,
}

ffi_conversions!(CartonAsyncNotifier);

/// This is a helper struct used to transport a result and arg over an mpsc queue
#[derive(Debug)]
pub struct ResultTransportWrapper {
    result: *mut c_void,
    status: CartonStatus,
    arg: CallbackArg,
}

unsafe impl Send for ResultTransportWrapper {}
unsafe impl Sync for ResultTransportWrapper {}

/// A struct that wraps a user provided arg and a notifier id
struct NotifierArgWrapper {
    inner: CallbackArg,
    notifier_id: u64,
}

ffi_conversions!(NotifierArgWrapper);

/// Notifier queues
static NOTIFIER_ID_GEN: AtomicU64 = AtomicU64::new(0);
static NOTIFIER_QUEUES: std::sync::OnceLock<
    DashMap<u64, mpsc::UnboundedSender<ResultTransportWrapper>>,
> = std::sync::OnceLock::new();

/// cbindgen:ignore
/// A callback that sends the result and arg on an mpsc queue that the user can later wait on
#[no_mangle]
pub extern "C" fn carton_async_notifier_internal_callback(
    result: *mut c_void,
    status: CartonStatus,
    arg: *mut c_void,
) {
    let wrapper: Box<NotifierArgWrapper> = (arg as *mut NotifierArgWrapper).into();
    let result = ResultTransportWrapper {
        result,
        status,
        arg: wrapper.inner,
    };

    let queue = NOTIFIER_QUEUES
        .get()
        .unwrap()
        .get(&wrapper.notifier_id)
        .unwrap();
    queue.send(result).unwrap();
}

impl CartonAsyncNotifier {
    /// Create an async notifier
    #[no_mangle]
    pub extern "C" fn carton_async_notifier_create(out: *mut *mut CartonAsyncNotifier) {
        let id = NOTIFIER_ID_GEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let (tx, rx) = mpsc::unbounded_channel();

        NOTIFIER_QUEUES
            .get_or_init(|| DashMap::new())
            .insert(id, tx);

        unsafe { *out = Box::new(CartonAsyncNotifier { id, recv: rx }).into() }
    }

    /// Wraps a user-provided arg and provides a callback for an async function call. See the docs on `CartonAsyncNotifier`
    /// for more detail.
    #[no_mangle]
    pub extern "C" fn carton_async_notifier_register(
        &self,
        callback_out: *mut CartonNotifierCallback,
        callback_arg: *mut CallbackArg,
    ) {
        unsafe {
            *callback_out = carton_async_notifier_internal_callback;

            // Wrap the user's callback arg with some other info
            let wrapper = Box::new(NotifierArgWrapper {
                inner: CallbackArg {
                    inner: (*callback_arg).inner,
                },
                notifier_id: self.id,
            });

            (*callback_arg).inner = Into::<*mut NotifierArgWrapper>::into(wrapper) as *mut _;
        }
    }

    /// Blocks until any async task is complete and returns the result in `result_out` and the user-provided arg in `callback_arg_out`
    /// Users are responsible for deleting `result_out` with the appropriate function (e.g. `carton_tensormap_destroy`)
    #[no_mangle]
    pub extern "C" fn carton_async_notifier_wait(
        &mut self,
        result_out: *mut *mut c_void,
        status: *mut CartonStatus,
        callback_arg_out: *mut CallbackArg,
    ) {
        let item = self.recv.blocking_recv().unwrap();

        unsafe {
            *result_out = item.result;
            *status = item.status;
            *callback_arg_out = item.arg;
        };
    }

    /// The same as `carton_async_notifier_wait`, but does not block if no async tasks are complete.
    /// Users are responsible for deleting `result_out` with the appropriate function (e.g. `carton_tensormap_destroy`)
    #[no_mangle]
    pub extern "C" fn carton_async_notifier_get(
        &mut self,
        result_out: *mut *mut c_void,
        status: *mut CartonStatus,
        callback_arg_out: *mut CallbackArg,
    ) -> CartonStatus {
        match self.recv.try_recv() {
            Ok(item) => unsafe {
                *result_out = item.result;
                *status = item.status;
                *callback_arg_out = item.arg;

                CartonStatus::Success
            },
            _ => unsafe {
                *result_out = std::ptr::null_mut();
                *callback_arg_out = CallbackArg {
                    inner: std::ptr::null_mut(),
                };

                CartonStatus::NoAsyncTasksReady
            },
        }
    }

    /// Destroy an async notifier
    #[no_mangle]
    pub extern "C" fn carton_async_notifier_destroy(notifier: *mut CartonAsyncNotifier) {
        let item: Box<CartonAsyncNotifier> = notifier.into();

        // Drop the send portion of the queue
        NOTIFIER_QUEUES.get().unwrap().remove(&item.id);

        drop(item)
    }
}
