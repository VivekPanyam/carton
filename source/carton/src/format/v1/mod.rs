//! This module implements v1 of the Carton file format spec
//! See `docs/specification/format.md` for more details
mod carton_toml;
pub(crate) mod links;
mod load;
mod tensor;
pub(crate) use load::load;

#[cfg(not(target_family = "wasm"))]
mod save;

#[cfg(not(target_family = "wasm"))]
pub(crate) use save::save;
