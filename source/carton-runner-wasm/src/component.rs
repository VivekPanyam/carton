wasmtime::component::bindgen!({
    world: "model",
    path: "./wit",
});

pub(crate) use carton_wasm::lib::types::{TensorNumeric, TensorString, Dtype};
use crate::component::carton_wasm::lib::types::Host;

pub(crate) struct DummyState;

impl Host for DummyState {}