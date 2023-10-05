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

use carton_core::types::{for_each_carton_type, for_each_numeric_carton_type, TypedStorage};
use ndarray::ShapeBuilder;
use std::ffi::{c_char, c_void, CStr};

use crate::types::{CartonStatus, DataType};

pub struct CartonTensor {
    inner: carton_core::types::Tensor,
    shape: Vec<u64>,
    strides: Vec<i64>,
}

ffi_conversions!(CartonTensor);

impl From<carton_core::types::Tensor> for CartonTensor {
    fn from(value: carton_core::types::Tensor) -> Self {
        for_each_carton_type! {
            return match &value {
                $(carton_core::types::Tensor::$CartonType(v) => {let v = v.view(); Self {shape: v.shape().iter().map(|v| (*v) as _).collect(), strides: v.strides().iter().map(|v| (*v) as _).collect(), inner: value}},)*
                carton_core::types::Tensor::NestedTensor(_) => panic!("Nested tensors are not yet supported in the C bindings!")
            }
        }
    }
}

impl From<Box<CartonTensor>> for carton_core::types::Tensor {
    fn from(value: Box<CartonTensor>) -> Self {
        value.inner
    }
}

/// A struct that lets us turn external blobs into `Tensor`s
struct ExternalBlobWrapper<T: 'static> {
    deleter: extern "C" fn(arg: *const c_void),
    deleter_arg: *const c_void,
    view: ndarray::ArrayViewMutD<'static, T>,
}

impl<T> TypedStorage<T> for ExternalBlobWrapper<T> {
    fn view(&self) -> ndarray::ArrayViewD<T> {
        self.view.view()
    }

    fn view_mut(&mut self) -> ndarray::ArrayViewMutD<T> {
        self.view.view_mut()
    }
}

impl<T> Drop for ExternalBlobWrapper<T> {
    fn drop(&mut self) {
        // TODO: null check on the deleter
        (self.deleter)(self.deleter_arg)
    }
}

impl CartonTensor {
    /// Create a numeric tensor by wrapping user-owned data.
    /// `deleter` will be called with `deleter_arg` when Carton no longer has references to `data`
    #[no_mangle]
    pub extern "C" fn carton_tensor_numeric_from_blob(
        data: *const c_void,
        dtype: DataType,
        shape: *const u64,
        strides: *const u64,
        num_dims: u64,
        deleter: extern "C" fn(arg: *const c_void),
        deleter_arg: *const c_void,
        tensor_out: *mut *mut CartonTensor,
    ) -> CartonStatus {
        let shape = unsafe { std::slice::from_raw_parts(shape, num_dims as _) };
        let strides = unsafe { std::slice::from_raw_parts(strides, num_dims as _) };

        for_each_numeric_carton_type! {
            match dtype {
                $(DataType::$CartonType => {
                    let view = unsafe {
                        ndarray::ArrayViewMutD::from_shape_ptr(
                            shape.into_iter().map(|v| (*v) as _).collect::<Vec<_>>().strides(strides.into_iter().map(|v| (*v) as _).collect()),
                            data as *mut $RustType
                        )
                    };
                    let ebr = ExternalBlobWrapper {
                        deleter,
                        deleter_arg,
                        view
                    };

                    let tensor = carton_core::types::Tensor::new(ebr);
                    let tensor = Box::new(CartonTensor::from(tensor));
                    unsafe { *tensor_out = tensor.into() };
                },)*
                _ => panic!("Only numeric tensors are supported here")
            }
        }

        CartonStatus::Success
    }

    /// Create a tensor with a provided shape and data type
    #[no_mangle]
    pub extern "C" fn carton_tensor_create(
        dtype: DataType,
        shape: *const u64,
        num_dims: u64,
        tensor_out: *mut *mut CartonTensor,
    ) {
        let shape = unsafe { std::slice::from_raw_parts(shape, num_dims as _) };

        for_each_carton_type! {
            match dtype {
                $(DataType::$CartonType => {
                    let item = ndarray::ArrayD::<$RustType>::default(shape.into_iter().map(|v| (*v) as _).collect::<Vec<_>>());

                    let tensor = carton_core::types::Tensor::new(item);
                    let tensor = Box::new(CartonTensor::from(tensor));
                    unsafe { *tensor_out = tensor.into() };
                },)*
            }
        }
    }

    /// Get a pointer to the underlying tensor data. This only works for numeric tensors.
    /// Sets `data_out` to NULL if not numeric.
    /// Note: the returned data pointer is only valid until this CartonTensor is destroyed.
    #[no_mangle]
    pub extern "C" fn carton_tensor_data(&self, data_out: *mut *mut c_void) {
        for_each_numeric_carton_type! {
            match &self.inner {
                $(carton_core::types::Tensor::$CartonType(v) => unsafe { *data_out = v.view().as_ptr() as _ },)*
                _ => unsafe {*data_out = std::ptr::null_mut()}
            }
        }
    }

    /// Get the datatype of a tensor.
    #[no_mangle]
    pub extern "C" fn carton_tensor_dtype(&self, dtype_out: *mut DataType) {
        for_each_carton_type! {
            return match &self.inner {
                $(carton_core::types::Tensor::$CartonType(_) => unsafe { *dtype_out = DataType::$CartonType },)*
                carton_core::types::Tensor::NestedTensor(_) => panic!("Nested tensors are not yet supported in the C bindings!")
            }
        }
    }

    /// Get the shape of a tensor.
    /// Note: the returned data pointer is only valid until this CartonTensor is destroyed.
    #[no_mangle]
    pub extern "C" fn carton_tensor_shape(
        &self,
        shape_out: *mut *const u64,
        num_dims_out: *mut u64,
    ) {
        unsafe {
            *num_dims_out = self.shape.len() as u64;
            *shape_out = self.shape.as_ptr();
        }
    }

    /// Get the strides of a tensor.
    /// Note: the returned data pointer is only valid until this CartonTensor is destroyed.
    #[no_mangle]
    pub extern "C" fn carton_tensor_strides(
        &self,
        strides_out: *mut *const i64,
        num_dims_out: *mut u64,
    ) {
        unsafe {
            *num_dims_out = self.strides.len() as u64;
            *strides_out = self.strides.as_ptr();
        }
    }

    /// For a string tensor, get a string at a particular (flattened) index into the tensor.
    /// Note: any returned pointers are only valid until the tensor is modified.
    #[no_mangle]
    pub extern "C" fn carton_tensor_get_string(
        &self,
        index: u64,
        string_out: *mut *const c_char,
        strlen_out: *mut u64,
    ) {
        if let carton_core::types::Tensor::String(v) = &self.inner {
            let view = v.view();
            let item = view.iter().nth(index as _).unwrap();
            unsafe {
                *string_out = item.as_ptr() as *const _;
                *strlen_out = item.len() as _;
            }
        } else {
            panic!("Tried to call `get_string` on a non-string tensor")
        }
    }

    /// For a string tensor, set a string at a particular (flattened) index.
    #[no_mangle]
    pub extern "C" fn carton_tensor_set_string(&mut self, index: u64, string: *const c_char) {
        let new = unsafe { CStr::from_ptr(string).to_str().unwrap().to_owned() };
        self.carton_tensor_set_string_inner(index, new);
    }

    #[no_mangle]
    pub extern "C" fn carton_tensor_set_string_with_strlen(
        &mut self,
        index: u64,
        string: *const c_char,
        strlen: u64,
    ) {
        let new = unsafe {
            std::str::from_utf8(std::slice::from_raw_parts(string as *const _, strlen as _))
                .unwrap()
                .to_owned()
        };

        self.carton_tensor_set_string_inner(index, new);
    }

    fn carton_tensor_set_string_inner(&mut self, index: u64, string: String) {
        if let carton_core::types::Tensor::String(v) = &mut self.inner {
            let mut view = v.view_mut();
            let item = view.iter_mut().nth(index as _).unwrap();
            *item = string;
        } else {
            panic!("Tried to call `set_string` on a non-string tensor")
        }
    }

    /// Destroy a CartonTensor
    #[no_mangle]
    pub extern "C" fn carton_tensor_destroy(tensor: *mut CartonTensor) {
        let item: Box<CartonTensor> = tensor.into();
        drop(item)
    }
}
