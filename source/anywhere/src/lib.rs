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

use lunchbox::types::{HasFileType, MaybeSend, MaybeSync};
pub use lunchbox::{ReadableFileSystem, WritableFileSystem};
use rpc::ServerBuilder;

mod file_ops;
pub mod path;
mod read_dir_ops;
pub mod rpc;
mod serialize;
pub mod transport;
pub mod types;

pub trait Servable<'a, const WRITABLE: bool, const SEEKABLE: bool>: HasFileType + Sized
where
    Self::FileType: MaybeSend + MaybeSync,
{
    // Every server must be at least readable so we set that to true
    fn build_server(&'a self) -> ServerBuilder<'a, Self, true /* READABLE */, WRITABLE, SEEKABLE> {
        ServerBuilder::new(&self)
    }
}

impl<T: HasFileType + Sized, const WRITABLE: bool, const SEEKABLE: bool>
    Servable<'_, WRITABLE, SEEKABLE> for T
where
    Self::FileType: MaybeSend + MaybeSync,
{
}
