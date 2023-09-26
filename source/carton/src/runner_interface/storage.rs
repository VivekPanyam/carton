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

//! Implementation of an enum that wraps all the runner tensor types and provides storage

use crate::types::{TensorStorage, TypedStorage};
use lunchbox::types::{MaybeSend, MaybeSync};

#[derive(Debug)]
pub struct RunnerStorage;

impl TensorStorage for RunnerStorage {
    type TypedStorage<T> = TypedRunnerStorage<T>
    where
        T: MaybeSend + MaybeSync;

    type TypedStringStorage = TypedRunnerStorage<String>;
}

#[derive(Debug)]
pub enum TypedRunnerStorage<T> {
    V1(runner_interface_v1::types::TensorStorage<T>),
}

impl<T> TypedStorage<T> for TypedRunnerStorage<T> {
    fn view(&self) -> ndarray::ArrayViewD<T> {
        match self {
            TypedRunnerStorage::V1(s) => s.view(),
        }
    }

    fn view_mut(&mut self) -> ndarray::ArrayViewMutD<T> {
        match self {
            TypedRunnerStorage::V1(s) => s.view_mut(),
        }
    }
}
