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

//! TensorStorage that is stored inline

use std::{fmt::Debug, marker::PhantomData};

use ndarray::{ShapeBuilder, StrideShape};
use serde::{Deserialize, Serialize};

use super::alloc::AsPtr;

#[derive(Debug, Serialize, Deserialize)]
pub struct TensorStorage<T, Storage> {
    pub(crate) data: Storage,
    pub(crate) shape: Vec<u64>,
    pub(crate) strides: Option<Vec<u64>>,
    pub(crate) pd: PhantomData<T>,
}

impl<T, Storage> TensorStorage<T, Storage>
where
    Storage: AsPtr<T>,
{
    fn get_shape(&self) -> StrideShape<ndarray::IxDyn> {
        match &self.strides {
            None => self
                .shape
                .iter()
                .map(|v| *v as usize)
                .collect::<Vec<_>>()
                .into(),
            Some(strides) => self
                .shape
                .iter()
                .map(|v| *v as usize)
                .collect::<Vec<_>>()
                .strides(strides.iter().map(|v| (*v).try_into().unwrap()).collect())
                .into(),
        }
    }

    pub fn view(&self) -> ndarray::ArrayViewD<T> {
        let data = self.data.as_ptr();
        unsafe { ndarray::ArrayView::from_shape_ptr(self.get_shape(), data) }
    }

    pub fn view_mut(&mut self) -> ndarray::ArrayViewMutD<T> {
        let data = self.data.as_mut_ptr();
        unsafe { ndarray::ArrayViewMut::from_shape_ptr(self.get_shape(), data) }
    }
}
