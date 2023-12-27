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
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{
    alloc::AllocatableBy,
    alloc_inline::{InlineAllocator, InlineTensorStorage},
    comms::Comms,
};

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct RPCRequest {
    pub id: RpcId,

    pub data: RPCRequestData,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct RPCResponse {
    pub id: RpcId,

    // Whether this is the final response for this request
    pub complete: bool,

    pub data: RPCResponseData,
}

pub(crate) type RpcId = u64;

// Used in multiplexer
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy, Serialize, Deserialize)]
pub(crate) struct StreamID(pub(crate) u64);

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy, Serialize, Deserialize)]
pub struct FsToken(pub(crate) StreamID);

// Individual channels/streams to avoid head of line blocking
#[allow(non_camel_case_types)]
#[allow(dead_code)]
#[repr(u8)]
pub(crate) enum ChannelId {
    Rpc = 0,
    FileSystem,
    CartonData,

    // Reserved
    NUM_RESERVED_IDS,
}

#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub(crate) struct FdId(pub(crate) u64);

// With this interface, creating a Carton is
//      Pack(data) -> core library packaging -> model.carton
// Loading it is
//      model.carton -> core library unpackaging -> Load(data)
//
// Loading an unpacked model is effectively
//      Pack(data) -> Load(data)
// (from the perspective of a runner)

#[derive(Debug, Serialize, Deserialize)]
pub(crate) enum RPCRequestData {
    Load {
        /// This filesystem points to a folder that is of the same structure as the output of `Pack` (for a particular runner)
        /// For a readonly filesystem
        fs: FsToken,

        runner_name: String,
        required_framework_version: semver::VersionReq,
        runner_compat_version: u64,
        runner_opts: Option<HashMap<String, RunnerOpt>>,
        visible_device: Device,

        // The hash of the model
        // This should always be avalable unless we're loading an unpacked model
        carton_manifest_hash: Option<String>,
    },

    // Pack a model
    Pack {
        /// A token for a read/write filesystem that the below paths reference
        fs: FsToken,

        // The path to user input data
        // If this is a folder, the runner is allowed to place data in a `.carton` subfolder
        // This can be used if it wants to generate a lockfile for example
        input_path: String,

        // A temporary folder generated by the core library. The runner can use this if it needs
        // to generate output in a new folder.
        // (In some cases, the input can be wrapped as-is and doesn't need to be copied into a new folder)
        // This folder is owned by the core library and will be deleted by it
        temp_folder: String,
    },

    Seal {
        tensors: HashMap<String, Handle<Tensor>>,
    },

    InferWithTensors {
        tensors: HashMap<String, Handle<Tensor>>,

        // Do we support a streaming response
        streaming: bool,
    },

    InferWithHandle {
        handle: SealHandle,

        // Do we support a streaming response
        streaming: bool,
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) enum RPCResponseData {
    // Doesn't return anything on successful load
    Load,

    Pack {
        // The path to the output directory. This can be in the temp folder passed into `Pack`
        // Note: this must be a *directory* even if the input was a file
        // This references a path on the FS that was passed in
        // during the request
        output_path: String,
    },

    Seal {
        handle: SealHandle,
    },

    Infer {
        tensors: HashMap<String, Handle<Tensor>>,
    },

    /// Something went wrong
    Error {
        e: String,
    },

    /// Logging
    LogMessage {
        record: LogRecord,
    },

    Empty,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LogRecord {
    metadata: LogMetadata,
    args: String,
    module_path: Option<String>,
    file: Option<String>,
    line: Option<u32>,
}

impl<'a> From<&log::Record<'a>> for LogRecord {
    fn from(value: &log::Record<'a>) -> Self {
        Self {
            metadata: value.metadata().into(),
            args: value.args().to_string(),
            module_path: value.module_path().map(|v| v.to_owned()),
            file: value.file().map(|v| v.to_owned()),
            line: value.line(),
        }
    }
}

impl LogRecord {
    /// Log to the currently active logger
    pub(crate) fn do_log(&self) {
        log::logger().log(
            &log::RecordBuilder::new()
                .level(self.metadata.level)
                .target(&self.metadata.target)
                .args(format_args!("{}", self.args))
                .module_path(self.module_path.as_ref().map(|v| v.as_str()))
                .file(self.file.as_ref().map(|v| v.as_str()))
                .line(self.line)
                .build(),
        );
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LogMetadata {
    level: log::Level,
    target: String,
}

impl<'a> From<&log::Metadata<'a>> for LogMetadata {
    fn from(value: &log::Metadata<'a>) -> Self {
        Self {
            level: value.level(),
            target: value.target().to_owned(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub enum RunnerOpt {
    Integer(i64),
    Double(f64),
    String(String),
    Boolean(bool),
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy, Serialize, Deserialize)]
pub(crate) struct SealHandle(pub(crate) u64);

#[derive(Debug, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Device {
    CPU,
    GPU {
        /// The UUID of the specified device
        /// This must include the `GPU-` or `MIG-GPU-` prefix
        /// See https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars
        uuid: Option<String>,
    },
}
// TODO: pin the version of carton-macros so we don't accidentally change the types on the wire
for_each_carton_type! {
    /// TODO: We should to manually implement serialization and not depend on ndarray's serialization
    /// staying the same. Or just pin to a specific ndarray version
    #[derive(Debug, Serialize, Deserialize)]
    pub enum Tensor {
        $($CartonType(TensorStorage::<$RustType>),)*

        /// A Nested Tensor / Ragged Tensor
        /// See the docs in the core carton library for more details
        NestedTensor(Vec<Tensor>)
    }
}

pub type TensorStorage<T> = super::storage::TensorStorage<T, InlineTensorStorage>;

pub trait Allocatable: AllocatableBy<InlineAllocator> {}
impl<T> Allocatable for T where T: AllocatableBy<InlineAllocator> {}

for_each_carton_type! {
    $(
        impl From<TensorStorage<$RustType>> for Tensor {
            fn from(item: TensorStorage<$RustType>) -> Self {
                Tensor::$CartonType(item)
            }
        }
    )*
}

// For now, we'll always serialize inline, but if we enable shared memory, we can handle that here
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct Handle<T> {
    inner: T,
}

impl Handle<Tensor> {
    pub(crate) async fn new(inner: Tensor, _comms: &Comms) -> Self {
        Self { inner }
    }

    pub(crate) async fn into_inner(self, _comms: &Comms) -> Tensor {
        self.inner
    }
}
