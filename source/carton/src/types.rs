pub use carton_macros::for_each_carton_type;
use lazy_static::lazy_static;
use lunchbox::types::{MaybeSend, MaybeSync};
use std::{collections::HashMap, fmt::Debug};

/// An opaque handle returned by `seal`
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct SealHandle(pub(crate) u64);

/// Options provided when loading a Carton
#[derive(Default)]
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
#[derive(Debug, Default, Clone)]
pub enum Device {
    #[default]
    CPU,
    GPU {
        /// The UUID of the specified device
        /// This must include the `GPU-` or `MIG-GPU-` prefix
        /// See https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars
        uuid: Option<String>,
    },
}

#[cfg(not(target_family = "wasm"))]
lazy_static! {
    static ref NVML: Option<nvml_wrapper::Nvml> = { nvml_wrapper::Nvml::init().ok() };
}

impl Device {
    #[cfg(target_family = "wasm")]
    pub fn maybe_from_str(s: &str) -> Self {
        if s.to_lowercase() == "cpu" {
            Device::CPU
        } else {
            Device::GPU { uuid: None }
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
    fn maybe_from_index(i: u32) -> Self {
        if let Some(nvml) = NVML.as_ref() {
            if let Ok(device) = nvml.device_by_index(i) {
                if let Ok(uuid) = device.uuid() {
                    return Self::GPU { uuid: Some(uuid) };
                }
            }
        }

        // Fall back to CPU
        // TODO: warn or throw an error
        Device::CPU
    }
}

/// Options that can be specified when packing a model
pub type PackOpts = CartonInfo;

// Directly expose everything in the carton.toml for now
// TODO: have an intermediate type
pub type CartonInfo = crate::info::CartonInfo;

for_each_carton_type! {
    /// The core tensor type
    pub enum Tensor {
        $($CartonType(NDarray::<$RustType>),)*

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

pub type NDarray<T> = runner_interface_v1::types::NDarray<T, GenericStorage<T>>;
pub type DataType = crate::info::DataType;

/// An alias for AsRef<[T]> + Debug + MaybeSend + MaybeSync
pub trait AsRefAndDebug<T>: AsRef<[T]> + Debug + MaybeSend + MaybeSync {}
impl<T, U> AsRefAndDebug<T> for U where U: AsRef<[T]> + Debug + MaybeSend + MaybeSync {}

#[derive(Debug)]
pub struct GenericStorage<T> {
    inner: Box<dyn AsRefAndDebug<T>>,
}

impl<T> GenericStorage<T> {
    pub fn new<U>(value: U) -> Self
    where
        U: AsRefAndDebug<T> + 'static,
    {
        Self {
            inner: Box::new(value),
        }
    }
}

impl<T> AsRef<[T]> for GenericStorage<T> {
    fn as_ref(&self) -> &[T] {
        self.inner.as_ref().as_ref()
    }
}
