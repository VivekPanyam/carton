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

use carton_macros::for_each_numeric_carton_type;

/// Numeric tensor types supported by this version of the runner interface
pub(crate) trait NumericTensorType: Default + Copy {}

for_each_numeric_carton_type! {
    $(
        impl NumericTensorType for $RustType {}
    )*
}

pub trait AsPtr<T> {
    /// Get a view of this tensor
    fn as_ptr(&self) -> *const T;

    /// Get a mut view of this tensor
    fn as_mut_ptr(&mut self) -> *mut T;
}

pub trait TypedAlloc<T> {
    type Output: AsPtr<T>;

    fn alloc(&self, numel: usize) -> Self::Output;
}
