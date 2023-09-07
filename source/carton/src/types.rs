pub use carton_macros::{for_each_carton_type, for_each_numeric_carton_type};
use serde::{de::Visitor, Deserialize, Serialize};
use std::collections::HashMap;

use crate::conversion_utils::{ConvertFromWithContext, ConvertIntoWithContext};
use lunchbox::types::{MaybeSend, MaybeSync};

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
pub type PackOpts<T> = CartonInfo<T>;

pub type CartonInfo<T> = crate::info::CartonInfo<T>;

for_each_numeric_carton_type! {
    /// The core tensor type
    #[derive(Debug)]
    pub enum Tensor<Storage> where Storage: TensorStorage {
        $($CartonType(Storage::TypedStorage::<$RustType>),)*

        String(Storage::TypedStringStorage),

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
        NestedTensor(Vec<Tensor<Storage>>)
    }
}

for_each_carton_type! {
    impl Clone for Tensor<GenericStorage> {
        fn clone(&self) -> Self {
            match self {
                $(
                    Self::$CartonType(item) => Self::$CartonType(item.clone()),
                )*
                Self::NestedTensor(item) => Self::NestedTensor(item.clone()),
            }
        }
    }
}

for_each_numeric_carton_type! {
    /// Implement conversions between tensor of different types
    impl<T, U, C> ConvertFromWithContext<Tensor<T>, C> for Tensor<U>
    where
        T: TensorStorage,
        U: TensorStorage,
        C: Copy,
        U::TypedStringStorage: ConvertFromWithContext<T::TypedStringStorage, C>,
        $(
            U::TypedStorage<$RustType>: ConvertFromWithContext<T::TypedStorage<$RustType>, C>,
        )*
    {
        fn from(item: Tensor<T>, context: C) -> Self {
            match item {
                $(
                    Tensor::$CartonType(item) => Self::$CartonType(item.convert_into_with_context(context)),
                )*
                Tensor::String(item) => Self::String(item.convert_into_with_context(context)),
                Tensor::NestedTensor(item) => Self::NestedTensor(item.convert_into_with_context(context))
            }
        }
    }
}

pub trait TensorStorage {
    /// Storage for each tensor type
    type TypedStorage<T>: TypedStorage<T> + MaybeSend + MaybeSync
    where
        T: MaybeSend + MaybeSync;

    type TypedStringStorage: TypedStorage<String> + MaybeSend + MaybeSync;
}

pub trait TypedStorage<T> {
    // Get a view of this tensor
    fn view(&self) -> ndarray::ArrayViewD<T>;

    // Get a mut view of this tensor
    fn view_mut(&mut self) -> ndarray::ArrayViewMutD<T>;
}

pub type DataType = crate::info::DataType;

pub struct GenericStorage;
impl TensorStorage for GenericStorage {
    type TypedStorage<T> = ndarray::ArrayD<T>  where T: MaybeSend + MaybeSync;
    type TypedStringStorage = ndarray::ArrayD<String>;
}

impl<T> TypedStorage<T> for ndarray::ArrayD<T> {
    fn view(&self) -> ndarray::ArrayViewD<T> {
        self.view()
    }

    fn view_mut(&mut self) -> ndarray::ArrayViewMutD<T> {
        self.view_mut()
    }
}

impl<Other, T, C> ConvertFromWithContext<Other, C> for ndarray::ArrayD<T>
where
    Other: TypedStorage<T>,
    T: Clone,
    C: Copy,
{
    fn from(value: Other, _context: C) -> Self {
        // TODO: refactor to improve performance
        value.view().to_owned()
    }
}
