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

pub(crate) mod alloc;
pub(crate) mod alloc_inline;
mod alloc_pool;
mod framed;
pub(crate) mod storage;
pub mod types;

if_not_wasm! {
    pub(crate) mod alloc_shm;
    pub(crate) mod comms;
}

if_wasm! {
    pub(crate) mod wasm_comms;
    pub(crate) use wasm_comms as comms;
}
