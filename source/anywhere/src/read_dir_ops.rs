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

use async_trait::async_trait;
use lunchbox::{types::MaybeSend, types::ReadableFile, ReadableFileSystem};
use serde::{Deserialize, Serialize};

use crate::types::RPCPath;

#[cfg_attr(target_family = "wasm", async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait)]
pub(crate) trait ReadDirOps: ReadableFileSystem
where
    Self::FileType: ReadableFile,
    Self::ReadDirPollerType: MaybeSend,
{
    async fn read_dir_wrapper(&self, path: RPCPath) -> std::io::Result<Vec<SerializedDirEntry>> {
        let mut out = Vec::new();
        let mut dir = self.read_dir(path).await.unwrap();
        while let Some(item) = dir.next_entry().await? {
            let path = item.path();
            let file_name = item.file_name();

            out.push(SerializedDirEntry {
                file_name,
                path: path.into_string(),
            });
        }

        Ok(out)
    }
}

impl<T> ReadDirOps for T
where
    T: ReadableFileSystem,
    T::FileType: ReadableFile,
    T::ReadDirPollerType: MaybeSend,
{
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SerializedDirEntry {
    pub(crate) file_name: String,
    pub(crate) path: String,
}
