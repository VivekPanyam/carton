//! Implements type conversions between carton::types and carton-runner-interface types for V1 of the runner interface

use std::fmt::Debug;

use crate::types::{Device, RunnerOpt, Tensor};
use crate::{
    conversion_utils::convert_vec,
    types::{GenericStorage, NDarray},
};
use carton_macros::for_each_numeric_carton_type;
use lunchbox::types::{MaybeSend, MaybeSync};
use runner_interface_v1::types::Storage;
use runner_interface_v1::types::TensorStorage;

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

fn convert_from_v1<T: Debug + MaybeSend + MaybeSync + 'static>(
    value: runner_interface_v1::types::NDarray<T, runner_interface_v1::types::TensorStorage<T>>,
) -> NDarray<T> {
    let shape = value.shape().clone();
    let strides = value.strides().clone();
    let storage = value.into_storage();

    #[derive(Debug)]
    struct Wrapper<T>(TensorStorage<T>);

    impl<T: Debug> AsRef<[T]> for Wrapper<T> {
        fn as_ref(&self) -> &[T] {
            self.0.get()
        }
    }

    let generic = GenericStorage::new(Wrapper(storage));
    NDarray::from_shape_strides_storage(shape, strides, generic)
}

// Implement conversions between tensor types
for_each_numeric_carton_type! {
    impl From<Tensor> for runner_interface_v1::types::Tensor {
        fn from(value: Tensor) -> Self {
            match value {
                $(
                    Tensor::$CartonType(v) => Self::$CartonType(v.into()),
                )*
                Tensor::NestedTensor(v) => Self::NestedTensor(convert_vec(v)),
                Tensor::String(_v) => panic!("String tensors aren't handled yet"),
            }
        }
    }

    impl From<runner_interface_v1::types::Tensor> for Tensor {
        fn from(value: runner_interface_v1::types::Tensor) -> Self {
            match value {
                $(
                    runner_interface_v1::types::Tensor::$CartonType(v) => Self::$CartonType(convert_from_v1(v)),
                )*
                runner_interface_v1::types::Tensor::NestedTensor(v) => Self::NestedTensor(convert_vec(v)),
                runner_interface_v1::types::Tensor::String(_v) => panic!("String tensors aren't handled yet"),
            }
        }
    }
}
