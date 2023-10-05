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

use std::{
    ffi::{c_char, CStr},
    sync::Arc,
};

use crate::{
    tensormap::CartonTensorMap,
    types::{CallbackArg, CartonInferCallback, CartonLoadCallback, CartonStatus},
    utils::runtime,
};

pub struct Carton {
    inner: Arc<carton_core::Carton>,
}

ffi_conversions!(Carton);

/// Load a carton
#[no_mangle]
pub extern "C" fn carton_load(
    url_or_path: *const c_char,
    callback: CartonLoadCallback,
    callback_arg: CallbackArg,
) {
    // Need to make a copy because we can only assume the string is valid until the function returns.
    let url_or_path = unsafe {
        CStr::from_ptr(url_or_path)
            .to_owned()
            .into_string()
            .unwrap()
    };

    carton_load_inner(url_or_path, callback, callback_arg);
}

/// Load a carton by providing a url and length
#[no_mangle]
pub extern "C" fn carton_load_with_strlen(
    url_or_path: *const c_char,
    strlen: u64,
    callback: CartonLoadCallback,
    callback_arg: CallbackArg,
) {
    // Need to make a copy because we can only assume the string is valid until the function returns.
    let url_or_path = unsafe {
        std::str::from_utf8(std::slice::from_raw_parts(
            url_or_path as *const _,
            strlen as _,
        ))
        .unwrap()
        .to_owned()
    };

    carton_load_inner(url_or_path, callback, callback_arg);
}

fn carton_load_inner(url_or_path: String, callback: CartonLoadCallback, callback_arg: CallbackArg) {
    // Spawn on the runtime
    runtime().spawn(async move {
        // TODO: expose LoadOpts
        // TODO: don't unwrap
        let inner = carton_core::Carton::load(url_or_path, carton_core::types::LoadOpts::default())
            .await
            .unwrap();
        let res = Box::new(Carton {
            inner: Arc::new(inner),
        });

        callback(res.into(), CartonStatus::Success, callback_arg.into());
    });
}

impl Carton {
    /// Run inference.
    /// Note: This function takes ownership of `tensors`.
    #[no_mangle]
    pub extern "C" fn carton_infer(
        &self,
        tensors: *mut CartonTensorMap,
        callback: CartonInferCallback,
        callback_arg: CallbackArg,
    ) {
        let inputs: Box<CartonTensorMap> = tensors.into();

        let carton = self.inner.clone();
        runtime().spawn(async move {
            // TODO: don't unwrap
            let inner = carton.infer(inputs.into_iter()).await.unwrap();

            let res: Box<CartonTensorMap> = Box::new(inner.into());

            callback(res.into(), CartonStatus::Success, callback_arg.into());
        });
    }

    /// Destroy a Carton
    #[no_mangle]
    pub extern "C" fn carton_destroy(carton: *mut Carton) {
        let item: Box<Carton> = carton.into();
        drop(item)
    }
}
