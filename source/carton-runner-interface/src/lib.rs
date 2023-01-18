//! This crate contains the interface between a runner and the core library
//! It is very important that no backward-incompatible changes are made within one major version of the runner interface
//! 
//! Each runner is built against one version of this crate.
//! The core library is built against *all* major versions of this crate

macro_rules! if_wasm {
    ($($item:item)*) => {$(
        #[cfg(target_family = "wasm")]
        $item
    )*}
}

macro_rules! if_not_wasm {
    ($($item:item)*) => {$(
        #[cfg(not(target_family = "wasm"))]
        $item
    )*}
}

mod do_not_modify;
mod client;
mod multiplexer;
mod runner;

if_not_wasm! {
    pub mod server;
}

pub use runner::Runner;
pub use do_not_modify::types;