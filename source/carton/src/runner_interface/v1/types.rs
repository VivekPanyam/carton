//! Implements type conversions between carton::types and carton-runner-interface types for V1 of the runner interface

use carton_macros::for_each_carton_type;

use crate::types::{Device, RunnerOpt, Tensor};


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
            RunnerOpt::Date(v) => Self::Date(v),
        }
    }
}

// Implement conversions between tensor types
for_each_carton_type! {
    impl From<Tensor> for runner_interface_v1::types::Tensor {
        fn from(value: Tensor) -> Self {
            match value {
                $(
                    Tensor::$CartonType(v) => Self::$CartonType(v),
                )*
            }
        }
    }

    impl From<runner_interface_v1::types::Tensor> for Tensor {
        fn from(value: runner_interface_v1::types::Tensor) -> Self {
            match value {
                $(
                    runner_interface_v1::types::Tensor::$CartonType(v) => Self::$CartonType(v),
                )*
            }
        }
    }
}