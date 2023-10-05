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

pub use carton_macros::{for_each_carton_type, for_each_numeric_carton_type};
use serde::{de::Visitor, Deserialize, Serialize};
use std::collections::HashMap;

/// An opaque handle returned by `seal`
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct SealHandle(pub(crate) u64);

/// Options provided when loading a Carton
#[derive(Default, Serialize, Deserialize)]
pub struct LoadOpts {
    /// Override the runner to use
    /// If not overridden, this is fetched from the carton metadata
    pub override_runner_name: Option<String>,

    /// Override the framework_version to use
    /// If not overridden, this is fetched from the carton metadata
    pub override_required_framework_version: Option<String>,

    /// Options to pass to the runner. These are runner-specific (e.g.
    /// PyTorch, TensorFlow, etc).
    ///
    /// Overrides are merged with the options set in the carton metadata
    /// Sometimes used to configure thread-pool sizes, etc.
    /// See the documentation for more info
    pub override_runner_opts: Option<HashMap<String, RunnerOpt>>,

    /// The device that is visible to this model.
    /// Note: a visible device does not necessarily mean that the model
    /// will use that device; it is up to the model to actually use it
    /// (e.g. by moving itself to GPU if it sees one available)
    pub visible_device: Device,
}

/// The types of options that can be passed to runners
pub type RunnerOpt = crate::info::RunnerOpt;

/// Supported device types
#[derive(Debug, Clone)]
pub enum Device {
    CPU,
    GPU {
        /// The UUID of the specified device
        /// This must include the `GPU-` or `MIG-GPU-` prefix
        /// See https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars
        uuid: Option<String>,
    },
}

/// Default to the first visible GPU (if any)
impl Default for Device {
    #[cfg(not(target_family = "wasm"))]
    fn default() -> Self {
        Device::maybe_from_index(0)
    }

    #[cfg(target_family = "wasm")]
    fn default() -> Self {
        Device::GPU { uuid: None }
    }
}

impl Device {
    #[cfg(target_family = "wasm")]
    pub fn maybe_from_str(s: &str) -> crate::error::Result<Self> {
        if s.to_lowercase() == "cpu" {
            Ok(Device::CPU)
        } else {
            Ok(Device::GPU { uuid: None })
        }
    }

    #[cfg(not(target_family = "wasm"))]
    pub fn maybe_from_str(s: &str) -> crate::error::Result<Self> {
        // Check if it's an index

        use crate::error::CartonError;
        if let Ok(index) = s.parse::<u32>() {
            return Ok(Self::maybe_from_index(index));
        }

        // Check if it's a cpu
        if s.to_lowercase() == "cpu" {
            return Ok(Device::CPU);
        }

        // Check if it's a UUID
        if s.starts_with("GPU-") || s.starts_with("MIG-GPU-") {
            return Ok(Device::GPU {
                uuid: Some(s.to_string()),
            });
        }

        // TODO: return an error
        Err(CartonError::InvalidDeviceFormat(s.to_string()))
    }

    #[cfg(not(target_family = "wasm"))]
    pub fn maybe_from_index(i: u32) -> Self {
        match crate::cuda::get_uuid_for_device(i) {
            Some(uuid) => Device::GPU { uuid: Some(uuid) },
            // Fall back to CPU
            None => Device::CPU,
        }
    }
}

impl ToString for Device {
    fn to_string(&self) -> String {
        match self {
            Device::CPU => "cpu".into(),
            Device::GPU { uuid } => uuid.as_ref().unwrap_or(&"gpu".into()).to_owned(),
        }
    }
}

impl Serialize for Device {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

struct DeviceDeserializeVisitor;
impl<'de> Visitor<'de> for DeviceDeserializeVisitor {
    type Value = Device;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a string that identifies a device")
    }

    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Device::maybe_from_str(v).map_err(|e| E::custom(e))
    }
}

impl<'de> Deserialize<'de> for Device {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_str(DeviceDeserializeVisitor)
    }
}

/// Options that can be specified when packing a model
pub type PackOpts = crate::info::PackOpts;

pub type CartonInfo = crate::info::CartonInfo;

for_each_numeric_carton_type! {
    /// The core tensor type
    #[derive(Clone)]
    pub enum Tensor {
        $($CartonType(GenericTensorStorage::<$RustType>),)*

        String(GenericTensorStorage::<String>),

        /// A Nested Tensor / Ragged Tensor
        /// Effectively a list of tensors. Most frameworks have constraints on what these tensors can
        /// be, but Carton itself doesn't enforce any constraints other than that a `NestedTensor` cannot
        /// contain `NestedTensor`s.
        ///
        /// The runner for each framework is responsible for returning an error if a NestedTensor does
        /// not meet the requirements for the framework.
        ///
        /// Torch only requires that the number of dimensions of the contained tensors are the same:
        /// https://pytorch.org/docs/1.13/nested.html
        ///
        /// TensorFlow requires that the number of dimensions and the type of each contained tensor
        /// is the same:
        /// https://www.tensorflow.org/guide/ragged_tensor#what_you_can_store_in_a_ragged_tensor
        NestedTensor(Vec<Tensor>)
    }
}

for_each_carton_type! {
    $(
        impl From<GenericTensorStorage<$RustType>> for Tensor {
            fn from(value: GenericTensorStorage<$RustType>) -> Self {
                Self::$CartonType(value)
            }
        }

        impl<S: TypedStorage<$RustType> + 'static> From<S> for GenericTensorStorage<$RustType> {
            fn from(value: S) -> Self {
                Self::new(value)
            }
        }
    )*
}

impl Tensor {
    pub fn new<T: 'static, S: TypedStorage<T>>(item: S) -> Self
    where
        GenericTensorStorage<T>: From<S>,
        Tensor: From<GenericTensorStorage<T>>,
    {
        GenericTensorStorage::from(item).into()
    }
}

for_each_carton_type! {
    impl std::fmt::Debug for Tensor {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                $(
                    Self::$CartonType(item) => f.debug_tuple(stringify!($CartonType)).field(&item.view()).finish(),
                )*
                Self::NestedTensor(item) => f.debug_tuple("NestedTensor").field(item).finish(),
            }
        }
    }
}

for_each_carton_type! {
    impl PartialEq for Tensor {
        fn eq(&self, other: &Tensor) -> bool {
            match (self, other) {
                $(
                    (Self::$CartonType(me), Tensor::$CartonType(other))  => me.view() == other.view(),
                )*
                (Self::NestedTensor(me), Tensor::NestedTensor(other)) => std::iter::zip(me, other).map(|(a, b)| a == b).all(|v| v),
                _ => false,
            }
        }
    }
}

pub trait TypedStorage<T> {
    // Get a view of this tensor
    fn view(&self) -> ndarray::ArrayViewD<T>;

    // Get a mut view of this tensor
    fn view_mut(&mut self) -> ndarray::ArrayViewMutD<T>;
}

pub type DataType = crate::info::DataType;

impl<T> TypedStorage<T> for ndarray::ArrayD<T> {
    fn view(&self) -> ndarray::ArrayViewD<T> {
        self.view()
    }

    fn view_mut(&mut self) -> ndarray::ArrayViewMutD<T> {
        self.view_mut()
    }
}

/// This helps us do type erasure with the Tensor storage type
pub struct GenericTensorStorage<T: 'static> {
    // SAFETY: 'static is okay because the data stays alive as long as _keepalive is around
    view: ndarray::ArrayViewMutD<'static, T>,

    // To ensure the underlying data stays alive while this tensor exists
    // Note: after creation, this has no dynamic dispatch overhead until destruction; we never call a method on _keepalive
    _keepalive: Box<dyn TypedStorage<T>>,
}

impl<T: Clone> Clone for GenericTensorStorage<T> {
    fn clone(&self) -> Self {
        // Make a copy
        let copy = self.view.to_owned();
        Self::new(copy)
    }
}

impl<T> GenericTensorStorage<T> {
    pub(crate) fn new<S: TypedStorage<T> + 'static>(value: S) -> Self {
        let mut v = Box::new(value);

        // SAFETY: it's safe to extend the lifetime of `view` because we ensure that the underlying data
        // does not get deallocated until `view` is no longer accessible.
        let view = unsafe { std::mem::transmute(v.view_mut()) };
        Self {
            view,
            _keepalive: v,
        }
    }

    pub fn view<'a>(&'a self) -> ndarray::ArrayViewD<'a, T> {
        self.view.view()
    }

    pub fn view_mut<'a>(&'a mut self) -> ndarray::ArrayViewMutD<'a, T> {
        self.view.view_mut()
    }
}

// TODO: explain why this is okay
unsafe impl<T: Send> Send for GenericTensorStorage<T> {}
unsafe impl<T: Sync> Sync for GenericTensorStorage<T> {}
