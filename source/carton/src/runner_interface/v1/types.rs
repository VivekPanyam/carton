//! Implements type conversions between carton::types and carton-runner-interface types for V1 of the runner interface

use crate::conversion_utils::convert_vec;
use carton_macros::for_each_carton_type;

use crate::types::{Device, GenericStorage, RunnerOpt, Tensor, TensorStorage, TypedStorage};

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
                    // TODO: this always makes a copy
                    Tensor::$CartonType(v) => Self::$CartonType(v.view().to_owned()),
                )*
                Tensor::NestedTensor(v) => Self::NestedTensor(convert_vec(v)),
            }
        }
    }

    impl From<runner_interface_v1::types::Tensor> for Tensor<GenericStorage> {
        fn from(value: runner_interface_v1::types::Tensor) -> Self {
            match value {
                $(
                    runner_interface_v1::types::Tensor::$CartonType(v) => Self::$CartonType(v),
                )*
                runner_interface_v1::types::Tensor::NestedTensor(v) => Self::NestedTensor(convert_vec(v)),
            }
        }
    }
}
