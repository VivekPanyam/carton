mod framed;
pub mod ndarray;
pub mod types;

if_not_wasm! {
    pub(crate) mod comms;
    pub(crate) mod shm_storage;
    pub(crate) use shm_storage as storage;
}

if_wasm! {
    pub(crate) mod wasm_comms;
    pub(crate) use wasm_comms as comms;
    pub(crate) mod inline_storage;
    pub(crate) use inline_storage as storage;
}
