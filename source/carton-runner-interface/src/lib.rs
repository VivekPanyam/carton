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

mod client;
mod do_not_modify;
mod multiplexer;
mod runner;

if_not_wasm! {
    pub mod server;
}

if_not_wasm! {
    pub(crate) use tokio::spawn as do_spawn;

    /// `Send` if not on wasm
    pub(crate) use Send as MaybeSend;

    /// `Sync` if not on wasm
    pub(crate) use Sync as MaybeSync;
}

if_wasm! {
    pub(crate) use tokio::task::spawn_local as do_spawn;

    /// `Send` if not on wasm
    pub(crate) trait MaybeSend {}
    impl <T> MaybeSend for T {}

    /// `Sync` if not on wasm
    pub(crate) trait MaybeSync {}
    impl <T> MaybeSync for T {}
}

pub use do_not_modify::types;
pub use runner::Runner;
