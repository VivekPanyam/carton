mod framed;
pub(crate) mod types;

if_not_wasm! {
    pub(crate) mod comms;
}

if_wasm! {
    pub(crate) mod wasm_comms;
    pub(crate) use wasm_comms as comms;
}