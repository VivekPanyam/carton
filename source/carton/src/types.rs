use std::collections::HashMap;
pub use carton_macros::for_each_carton_type;
use serde::{Serialize, Deserialize};

/// TODO: this structure actually serializes and deserializes the tensors
/// to send back and forth. They should be put in shared memory

#[derive(Debug, Serialize, Deserialize)]
pub struct RPCRequest {
    pub id: RpcId,

    pub data: RPCRequestData,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RPCResponse {
    pub id: RpcId,

    pub data: RPCResponseData,
}

pub type RpcId = u64;

#[derive(Debug, Serialize, Deserialize)]
pub enum RPCRequestData {
    Load {
        /// The path to the model to load
        path: String,

        /// The runner to load the model with
        runner: Option<String>,

        /// The runner version to use
        runner_version: Option<String>,

        /// The options to pass to the runner
        runner_opts: Option<String>,

        /// The device that is visible to the runner
        visible_device: Device,
    },

    Seal {
        tensors: HashMap<String, Tensor>
    },

    InferWithTensors {
        tensors: HashMap<String, Tensor>
    },

    InferWithHandle {
        handle: SealHandle
    },
}

#[derive(Debug, Serialize, Deserialize)]
pub enum RPCResponseData {
    Load {
        name: String,
        
        runner: String,

        inputs: Vec<TensorSpec>,

        outputs: Vec<TensorSpec>,
    },

    Seal {
        handle: SealHandle
    },

    Infer {
        tensors: HashMap<String, Tensor>
    },

    /// Something went wrong
    Error {
        e: String,
    }

}

/// TODO: newtype?
pub type SealHandle = u64;

pub struct Schema {
    schema_type: SchemaType
}

pub enum SchemaType {
    /// We don't know what the schema for this model looks like
    Unknown,

    /// A schema was provided when the carton was generated
    UserProvided,

    /// Extracted from the model
    /// Generally, these schemas won't be as precise as user-specified ones
    FromModel,

}

#[derive(Debug, Serialize, Deserialize)]
pub enum Device {
    CPU,
    GPU {
        /// The UUID of the specified device
        uuid: Option<String>
    }
}

// TODO handle scalars

#[derive(Debug, Serialize, Deserialize)]
pub struct TensorSpec {
    pub name: String,
    pub dims: Vec<Dimension>,
    pub tensor_type: TensorType
}

/// A dimension can be either a fixed value, a symbol, or any value
#[derive(Debug, Serialize, Deserialize)]
pub enum Dimension {
    Value { value: u64 },
    Symbol { symbol: String},
    Any,
}


for_each_carton_type! {
    #[derive(Debug, Serialize, Deserialize)]
    pub enum TensorType {
        $($CartonType,)*
    }
}

for_each_carton_type! {
    #[derive(Debug, Serialize, Deserialize)]
    pub enum Tensor {
        $($CartonType(ndarray::ArrayD::<$RustType>),)*
    }
}

for_each_carton_type! {
    $(
        impl From<ndarray::ArrayD<$RustType>> for Tensor {
            fn from(item: ndarray::ArrayD<$RustType>) -> Self {
                Tensor::$CartonType(item)
            }
        }
    )*
}