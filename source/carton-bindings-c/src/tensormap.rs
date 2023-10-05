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
    collections::HashMap,
    ffi::{c_char, CStr},
};

use crate::tensor::CartonTensor;

/// A map from String to Tensor
/// This is the input and output type of `carton_infer`
pub struct CartonTensorMap {
    inner: HashMap<String, Box<CartonTensor>>,
}

ffi_conversions!(CartonTensorMap);

impl From<HashMap<String, carton_core::types::Tensor>> for CartonTensorMap {
    fn from(value: HashMap<String, carton_core::types::Tensor>) -> Self {
        let inner = value
            .into_iter()
            .map(|(k, v)| (k, Box::new(v.into())))
            .collect();
        Self { inner }
    }
}

impl IntoIterator for CartonTensorMap {
    type Item = (String, carton_core::types::Tensor);

    type IntoIter = std::iter::Map<
        std::collections::hash_map::IntoIter<String, Box<CartonTensor>>,
        fn((String, Box<CartonTensor>)) -> (String, carton_core::types::Tensor),
    >;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter().map(|(k, v)| (k, v.into()))
    }
}

impl CartonTensorMap {
    /// Create a CartonTensorMap
    #[no_mangle]
    pub extern "C" fn carton_tensormap_create(out: *mut *mut CartonTensorMap) {
        let res = Box::new(Self {
            inner: Default::default(),
        });

        unsafe { *out = res.into() };
    }

    /// Get a tensor from the map by name. `value_out` is set to null if there was no value for `key`
    /// This transfers ownership of `value_out` to the caller.
    #[no_mangle]
    pub extern "C" fn carton_tensormap_get_and_remove(
        &mut self,
        key: *const c_char,
        value_out: *mut *mut CartonTensor,
    ) {
        let key = unsafe { CStr::from_ptr(key).to_str().unwrap() };
        self.carton_tensormap_get_and_remove_inner(key, value_out);
    }

    /// Get a tensor from the map by name. `value_out` is set to null if there was no value for `key`
    /// This transfers ownership of `value_out` to the caller.
    #[no_mangle]
    pub extern "C" fn carton_tensormap_get_and_remove_with_strlen(
        &mut self,
        key: *const c_char,
        strlen: u64,
        value_out: *mut *mut CartonTensor,
    ) {
        let key = unsafe {
            std::str::from_utf8(std::slice::from_raw_parts(key as *const _, strlen as _)).unwrap()
        };
        self.carton_tensormap_get_and_remove_inner(key, value_out);
    }

    fn carton_tensormap_get_and_remove_inner(
        &mut self,
        key: &str,
        value_out: *mut *mut CartonTensor,
    ) {
        match self.inner.remove(key) {
            Some(v) => unsafe { *value_out = v.into() },
            None => unsafe { *value_out = std::ptr::null_mut() },
        }
    }

    /// Insert a tensor into the map.
    /// This function takes ownership of `value`.
    #[no_mangle]
    pub extern "C" fn carton_tensormap_insert(
        &mut self,
        key: *const c_char,
        value: *mut CartonTensor,
    ) {
        let key = unsafe { CStr::from_ptr(key).to_str().unwrap() };
        self.carton_tensormap_insert_inner(key, value);
    }

    /// Insert a tensor into the map.
    /// This function takes ownership of `value`.
    #[no_mangle]
    pub extern "C" fn carton_tensormap_insert_with_strlen(
        &mut self,
        key: *const c_char,
        strlen: u64,
        value: *mut CartonTensor,
    ) {
        let key = unsafe {
            std::str::from_utf8(std::slice::from_raw_parts(key as *const _, strlen as _)).unwrap()
        };
        self.carton_tensormap_insert_inner(key, value);
    }

    fn carton_tensormap_insert_inner(&mut self, key: &str, value: *mut CartonTensor) {
        self.inner.insert(key.to_owned(), value.into());
    }

    /// Get the number of elements in the map
    #[no_mangle]
    pub extern "C" fn carton_tensormap_len(&self, count_out: *mut u64) {
        unsafe { *count_out = self.inner.len() as _ };
    }

    // /// Get all keys and values into arrays
    // /// SAFETY: the returned data is only valid as long as the map is not modified
    // #[no_mangle]
    // pub extern "C" fn carton_tensormap_items(
    //     &self,
    //     max_count: u64,
    //     keys_array: *mut *const c_char,
    //     values_array: *mut *const CartonTensor,
    // ) {
    //     todo!()
    // }

    /// Destroy a CartonTensorMap
    #[no_mangle]
    pub extern "C" fn carton_tensormap_destroy(map: *mut CartonTensorMap) {
        let item: Box<CartonTensorMap> = map.into();
        drop(item)
    }
}
