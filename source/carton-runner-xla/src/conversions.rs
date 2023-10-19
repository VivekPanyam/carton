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

use carton_runner_interface::types::{
    for_each_numeric_carton_type, Allocatable, Tensor, TensorStorage,
};
use xla::{ArrayElement, ElementType, Literal};

pub(crate) fn tensor_to_literal(value: Tensor) -> Literal {
    for_each_numeric_carton_type! {
        match value {
            $(Tensor::$CartonType(v) => return storage_to_literal(v),)*
            Tensor::String(_) => panic!("String tensors are not currently supported by the XLA runner"),
            Tensor::NestedTensor(_) => panic!("Nested tensors are not currently supported by the XLA runner"),
        }
    }
}

fn storage_to_literal<T: ArrayElement>(storage: TensorStorage<T>) -> Literal {
    let view = storage.view();
    let data = view.as_slice().unwrap();
    let shape = view.shape();
    let ty = T::TY;

    // Convert to a u8 slice
    let untyped_data = unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<T>(),
        )
    };

    Literal::create_from_shape_and_untyped_data(ty, shape, untyped_data).unwrap()
}

pub(crate) fn literal_to_tensor(value: Literal) -> Tensor {
    match value.element_type().unwrap() {
        ElementType::S8 => literal_to_storage::<i8>(value).into(),
        ElementType::S16 => literal_to_storage::<i16>(value).into(),
        ElementType::S32 => literal_to_storage::<i32>(value).into(),
        ElementType::S64 => literal_to_storage::<i64>(value).into(),
        ElementType::U8 => literal_to_storage::<u8>(value).into(),
        ElementType::U16 => literal_to_storage::<u16>(value).into(),
        ElementType::U32 => literal_to_storage::<u32>(value).into(),
        ElementType::U64 => literal_to_storage::<u64>(value).into(),
        ElementType::F32 => literal_to_storage::<f32>(value).into(),
        ElementType::F64 => literal_to_storage::<f64>(value).into(),
        other => panic!("XLA tensor type {other:?} is not supported!"),
    }
}

pub fn literal_to_storage<T: Default + Clone + ArrayElement + Allocatable>(
    value: Literal,
) -> TensorStorage<T> {
    let shape_info = value.array_shape().unwrap();
    let mut out = TensorStorage::new(shape_info.dims().into_iter().map(|v| (*v) as _).collect());

    // Copy the data in
    let mut view = out.view_mut();
    let data = view.as_slice_mut().unwrap();
    value.copy_raw_to(data).unwrap();

    out
}
