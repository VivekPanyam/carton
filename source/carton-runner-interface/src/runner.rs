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

use std::{collections::HashMap, sync::Arc};

use crate::{
    client::Client,
    do_not_modify::comms::OwnedComms,
    do_not_modify::{
        alloc::TypedAlloc,
        alloc_inline::{InlineAllocator, InlineTensorStorage},
        types::{Device, RPCRequestData, RPCResponseData, SealHandle, Tensor},
    },
    types::{Handle, RunnerOpt, TensorStorage},
};

use futures::Stream;
use lunchbox::types::{MaybeSend, MaybeSync};

pub struct Runner {
    client: Client,
}

impl Runner {
    #[cfg(not(target_family = "wasm"))]
    pub async fn new(
        runner_path: &std::path::Path,
        visible_device: Device,
    ) -> Result<Runner, String> {
        use tokio::process::Command;

        // Make sure the runner exists
        if !runner_path.exists() {
            return Err("Runner doesn't exist".into());
        }

        // Create comms
        let (comms, uds_path) = OwnedComms::new().await;

        // Create a command to start the runner
        let mut command = Command::new(runner_path);

        // Check if we have a UUID for a GPU
        if let Device::GPU { uuid: Some(uuid) } = visible_device {
            // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars
            command.env("CUDA_VISIBLE_DEVICES", uuid);
        } else {
            // Hide all GPUs
            command.env("CUDA_VISIBLE_DEVICES", "");
        }

        command
            .args(["--uds-path", uds_path.to_str().unwrap()])
            .spawn()
            .expect("Runner failed to start");

        // Create a client
        let client = Client::new(comms).await;

        Ok(Self { client })
    }

    #[cfg(target_family = "wasm")]
    pub async fn new() -> Result<Runner, String> {
        // Create comms
        let comms = OwnedComms::new().await;

        // Create a client
        let client = Client::new(comms).await;

        Ok(Self { client })
    }

    pub async fn load<T>(
        &self,
        fs: &Arc<T>,
        runner_name: String,
        required_framework_version: semver::VersionReq,
        runner_compat_version: u64,
        runner_opts: Option<HashMap<String, RunnerOpt>>,
        visible_device: Device,
        carton_manifest_hash: Option<String>,
    ) -> Result<(), String>
    where
        T: lunchbox::ReadableFileSystem + MaybeSend + MaybeSync + 'static,
        T::FileType: lunchbox::types::ReadableFile + MaybeSend + MaybeSync + Unpin,
        T::ReadDirPollerType: MaybeSend,
    {
        // Serve the filesystem
        let token = self.client.serve_readonly_fs(fs.clone()).await;

        match self
            .client
            .do_rpc(RPCRequestData::Load {
                fs: token,
                runner_name,
                required_framework_version,
                runner_compat_version,
                runner_opts,
                visible_device,
                carton_manifest_hash,
            })
            .await
        {
            RPCResponseData::Load => Ok(()),
            RPCResponseData::Error { e } => Err(e),
            _ => panic!("Unexpected RPC response type!"),
        }
    }

    // pub async fn seal(&self, tensors: HashMap<String, Tensor>) -> Result<SealHandle, String> {
    //     match self.client.do_rpc(RPCRequestData::Seal { tensors }).await {
    //         RPCResponseData::Seal { handle } => Ok(handle),
    //         RPCResponseData::Error { e } => Err(e),
    //         _ => panic!("Unexpected RPC response type!"),
    //     }
    // }

    pub async fn infer_with_inputs(
        &self,
        tensors_orig: HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>, String> {
        // Wrap each tensor in a handle (this possibly sends the fd for backing SHM chunks to the other process)
        let comms = self.client.get_comms();
        let mut tensors = HashMap::new();
        for (k, v) in tensors_orig.into_iter() {
            tensors.insert(k, Handle::new(v, comms).await);
        }

        match self
            .client
            .do_rpc(RPCRequestData::InferWithTensors {
                tensors,
                streaming: false,
            })
            .await
        {
            RPCResponseData::Infer { tensors } => {
                let mut out = HashMap::new();
                for (k, v) in tensors.into_iter() {
                    out.insert(k, v.into_inner(comms).await);
                }

                Ok(out)
            }
            RPCResponseData::Error { e } => Err(e),
            _ => panic!("Unexpected RPC response type!"),
        }
    }

    pub async fn streaming_infer_with_inputs(
        &self,
        tensors_orig: HashMap<String, Tensor>,
    ) -> impl Stream<Item = Result<HashMap<String, Tensor>, String>> + '_ {
        // Wrap each tensor in a handle (this possibly sends the fd for backing SHM chunks to the other process)
        let comms = self.client.get_comms();
        let mut tensors = HashMap::new();
        for (k, v) in tensors_orig.into_iter() {
            tensors.insert(k, Handle::new(v, comms).await);
        }

        let mut res = self
            .client
            .do_streaming_rpc(RPCRequestData::InferWithTensors {
                tensors,
                streaming: true,
            })
            .await;

        async_stream::stream! {
            while let Some(v) = res.recv().await {
                match v {
                    RPCResponseData::Infer { tensors } => {
                        let mut out = HashMap::new();
                        for (k, v) in tensors.into_iter() {
                            out.insert(k, v.into_inner(comms).await);
                        }

                        yield Ok(out)
                    }
                    RPCResponseData::Error { e } => yield Err(e),
                    RPCResponseData::Empty => { } // We can get this on the last message. Do nothing
                    _ => panic!("Unexpected RPC response type!"),
                }
            }
        }
    }

    pub async fn seal(&self, tensors_orig: HashMap<String, Tensor>) -> Result<u64, String> {
        // Wrap each tensor in a handle (this possibly sends the fd for backing SHM chunks to the other process)
        let comms = self.client.get_comms();
        let mut tensors = HashMap::new();
        for (k, v) in tensors_orig.into_iter() {
            tensors.insert(k, Handle::new(v, comms).await);
        }

        match self.client.do_rpc(RPCRequestData::Seal { tensors }).await {
            RPCResponseData::Seal { handle } => Ok(handle.0),
            RPCResponseData::Error { e } => Err(e),
            _ => panic!("Unexpected RPC response type!"),
        }
    }

    pub async fn infer_with_handle(&self, handle: u64) -> Result<HashMap<String, Tensor>, String> {
        let comms = self.client.get_comms();

        match self
            .client
            .do_rpc(RPCRequestData::InferWithHandle {
                handle: SealHandle(handle),
                streaming: false,
            })
            .await
        {
            RPCResponseData::Infer { tensors } => {
                let mut out = HashMap::new();
                for (k, v) in tensors.into_iter() {
                    out.insert(k, v.into_inner(comms).await);
                }

                Ok(out)
            }
            RPCResponseData::Error { e } => Err(e),
            _ => panic!("Unexpected RPC response type!"),
        }
    }

    pub async fn streaming_infer_with_handle(
        &self,
        handle: u64,
    ) -> impl Stream<Item = Result<HashMap<String, Tensor>, String>> + '_ {
        let comms = self.client.get_comms();

        let mut res = self
            .client
            .do_streaming_rpc(RPCRequestData::InferWithHandle {
                handle: SealHandle(handle),
                streaming: true,
            })
            .await;

        async_stream::stream! {
            while let Some(v) = res.recv().await {
                match v {
                    RPCResponseData::Infer { tensors } => {
                        let mut out = HashMap::new();
                        for (k, v) in tensors.into_iter() {
                            out.insert(k, v.into_inner(comms).await);
                        }

                        yield Ok(out)
                    }
                    RPCResponseData::Error { e } => yield Err(e),
                    RPCResponseData::Empty => { } // We can get this on the last message. Do nothing
                    _ => panic!("Unexpected RPC response type!"),
                }
            }
        }
    }

    /// Pack a model and return a path to the output directory
    pub async fn pack<T>(
        &self,
        fs: &Arc<T>,
        input_path: &lunchbox::path::Path,
        temp_folder: &lunchbox::path::Path,
    ) -> Result<lunchbox::path::PathBuf, String>
    where
        T: lunchbox::WritableFileSystem + MaybeSend + MaybeSync + 'static,
        T::FileType: lunchbox::types::WritableFile + MaybeSend + MaybeSync + Unpin,
        T::ReadDirPollerType: MaybeSend,
    {
        // Serve the filesystem
        let token = self.client.serve_writable_fs(fs.clone()).await;

        match self
            .client
            .do_rpc(RPCRequestData::Pack {
                fs: token,
                input_path: input_path.to_string(),
                temp_folder: temp_folder.to_string(),
            })
            .await
        {
            RPCResponseData::Pack { output_path } => Ok(output_path.into()),
            RPCResponseData::Error { e } => Err(e),
            _ => panic!("Unexpected RPC response type!"),
        }
    }

    pub fn alloc_tensor<T: Clone + Default>(&self, shape: Vec<u64>) -> Result<Tensor, String>
    where
        InlineAllocator: TypedAlloc<T, Output = InlineTensorStorage>,
        Tensor: From<TensorStorage<T>>,
    {
        Ok(TensorStorage::new(shape).into())
    }

    // pub async fn infer_with_handle(
    //     &self,
    //     handle: SealHandle,
    // ) -> Result<HashMap<String, Tensor>, String> {
    //     match self
    //         .client
    //         .do_rpc(RPCRequestData::InferWithHandle { handle })
    //         .await
    //     {
    //         RPCResponseData::Infer { tensors } => Ok(tensors),
    //         RPCResponseData::Error { e } => Err(e),
    //         _ => panic!("Unexpected RPC response type!"),
    //     }
    // }
}
