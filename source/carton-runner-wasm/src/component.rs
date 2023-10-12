wasmtime::component::bindgen!({
    world: "model",
    path: "./wit",
});

use crate::component::carton_wasm::lib::types::Host;
pub(crate) use carton_wasm::lib::types::{Dtype, TensorNumeric, TensorString};

pub(crate) struct HostImpl;

impl Host for HostImpl {}
