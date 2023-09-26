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

//! Implements type conversions between carton::types and carton-runner-interface types for V1 of the runner interface

use crate::conversion_utils::convert_vec;
use carton_macros::for_each_carton_type;

use crate::runner_interface::storage::{RunnerStorage, TypedRunnerStorage};
use crate::types::{Device, RunnerOpt, Tensor, TensorStorage, TypedStorage};

impl From<Device> for runner_interface_v1::types::Device {
    fn from(value: Device) -> Self {
        match value {
            Device::CPU => Self::CPU,
            Device::GPU { uuid } => Self::GPU { uuid },
        }
    }
}

impl From<RunnerOpt> for runner_interface_v1::types::RunnerOpt {
    fn from(value: RunnerOpt) -> Self {
        match value {
            RunnerOpt::Integer(v) => Self::Integer(v),
            RunnerOpt::Double(v) => Self::Double(v),
            RunnerOpt::String(v) => Self::String(v),
            RunnerOpt::Boolean(v) => Self::Boolean(v),
        }
    }
}

// Implement conversions between tensor types
for_each_carton_type! {
    impl<T> From<Tensor<T>> for runner_interface_v1::types::Tensor where T: TensorStorage {
        fn from(value: Tensor<T>) -> Self {
            match value {
                $(
                    Tensor::$CartonType(v) => Self::$CartonType(v.view().into()),
                )*
                Tensor::NestedTensor(v) => Self::NestedTensor(convert_vec(v)),
            }
        }
    }

    impl From<runner_interface_v1::types::Tensor> for Tensor<RunnerStorage> {
        fn from(value: runner_interface_v1::types::Tensor) -> Self {
            match value {
                $(
                    // TODO: this makes a copy
                    runner_interface_v1::types::Tensor::$CartonType(v) => Self::$CartonType(TypedRunnerStorage::V1(v)),
                )*
                runner_interface_v1::types::Tensor::NestedTensor(v) => Self::NestedTensor(convert_vec(v)),
            }
        }
    }
}
