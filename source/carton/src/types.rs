pub use carton_macros::for_each_carton_type;
use uuid::Uuid;
use std::collections::HashMap;
use lazy_static::lazy_static;

/// An opaque handle returned by `seal`
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct SealHandle(u64);

/// Options provided when loading a Carton
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
pub type RunnerOpt = crate::format::v1::carton_toml::RunnerOpt;

/// Supported device types
pub enum Device {
    CPU,
    GPU {
        /// The UUID of the specified device
        uuid: Option<Uuid>,
    },
}

lazy_static! {
    static ref NVML: Option<nvml_wrapper::Nvml> = {
        nvml_wrapper::Nvml::init().ok()
    };
}

impl Device {
    pub fn maybe_from_str(s: &str) -> Self {
        // Check if it's an index
        if let Ok(index) = s.parse::<u32>() {
            return Self::maybe_from_index(index)
        }

        // Check if it's a cpu
        if s.to_lowercase() == "cpu" {
            return Device::CPU
        }

        // Check if it's a UUID
        if let Ok(u) = uuid::Uuid::try_parse(s) {
            return Device::GPU { uuid: Some(u)}
        }

        // TODO: return an error
        panic!("Invalid format for device. Expected `cpu`, a device index, or a UUID")

    }
    fn maybe_from_index(i: u32) -> Self {
        if let Some(nvml) = NVML.as_ref() {
            if let Ok(device) = nvml.device_by_index(i) {
                if let Ok(uuid) = device.uuid() {
                    // TODO: don't unwrap
                    return Self::GPU { uuid: Some(Uuid::try_parse(&uuid).unwrap()) }
                }
            }
        }

        // Fall back to CPU
        // TODO: warn or throw an error
        Device::CPU
    }
}

/// Options that can be specified when packing a model
// TODO: add options so we can fill everything in carton.toml
pub struct PackOpts {
    /// The name of the model
    model_name: Option<String>,

    /// The model description
    model_description: Option<String>,

    /// A list of platforms this model supports
    /// If empty or unspecified, all platforms are okay
    required_platforms: Option<Vec<target_lexicon::Triple>>,

    /// The name of the runner to use
    runner_name: String,

    /// The required framework version range to run the model with
    /// This is a semver version range. See https://docs.rs/semver/1.0.16/semver/struct.VersionReq.html
    /// for format details.
    /// For example `=1.2.4`, means exactly version `1.2.4`
    required_framework_version_range: semver::VersionReq,

    /// Don't set this unless you know what you're doing
    runner_compat_version: Option<u64>,

    /// Options to pass to the runner. These are runner-specific (e.g.
    /// PyTorch, TensorFlow, etc).
    ///
    /// Sometimes used to configure thread-pool sizes, etc.
    /// See the documentation for more info
    opts: Option<HashMap<String, RunnerOpt>>,
}

// Directly expose everything in the carton.toml for now
// TODO: have an intermediate type
pub type CartonInfo = crate::format::v1::carton_toml::CartonToml;

for_each_carton_type! {
    /// The core tensor type
    pub enum Tensor {
        $($CartonType(ndarray::ArrayD::<$RustType>),)*

        // A Nested Tensor / Ragged Tensor
        // NestedTensor(Vec<Tensor>)
    }
}

for_each_carton_type! {
    $(
        /// Implement conversions from ndarray types
        impl From<ndarray::ArrayD<$RustType>> for Tensor {
            fn from(item: ndarray::ArrayD<$RustType>) -> Self {
                Tensor::$CartonType(item)
            }
        }
    )*
}
