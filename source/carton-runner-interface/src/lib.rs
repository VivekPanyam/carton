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
pub mod runner;

if_not_wasm! {
    pub mod server;
    pub mod slowlog;
}

if_not_wasm! {
    pub(crate) use tokio::spawn as do_spawn;

    /// `Send` if not on wasm
    pub(crate) use Send as MaybeSend;
}

if_wasm! {
    pub(crate) use tokio::task::spawn_local as do_spawn;

    /// `Send` if not on wasm
    pub(crate) trait MaybeSend {}
    impl <T> MaybeSend for T {}
}

pub use do_not_modify::types;
pub use runner::Runner;

#[cfg(feature = "benchmark")]
pub mod _only_public_for_benchmarks_do_not_use {
    pub use crate::do_not_modify::alloc::TypedAlloc;
    pub use crate::do_not_modify::alloc_inline::{
        alloc_tensor as alloc_tensor_inline, alloc_tensor_no_pool as alloc_tensor_no_pool_inline,
        InlineAllocator, InlineTensorStorage,
    };

    pub use crate::do_not_modify::alloc_shm::{
        alloc_tensor as alloc_tensor_shm, alloc_tensor_no_pool as alloc_tensor_no_pool_shm,
        SHMAllocator, SHMTensorStorage,
    };
}
