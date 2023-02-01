pub(crate) mod alloc;
pub(crate) mod alloc_inline;
mod alloc_pool;
mod framed;
pub(crate) mod storage;
pub mod types;

if_not_wasm! {
    pub(crate) mod comms;
}

if_wasm! {
    pub(crate) mod wasm_comms;
    pub(crate) use wasm_comms as comms;
}
