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

use std::collections::HashMap;

use carton_runner_interface::{
    server::{init_runner, RequestData, ResponseData, SealHandle},
    types::Tensor,
};
use lunchbox::{path::PathBuf, ReadableFileSystem};
use serde::{Deserialize, Serialize};
use xla::PjRtLoadedExecutable;

use crate::conversions::{literal_to_tensor, tensor_to_literal};

mod conversions;

#[tokio::main]
async fn main() {
    // On x86 linux, we need to set up the environment to get cuda to work
    #[cfg(all(not(target_os = "macos"), target_arch = "x86_64"))]
    if std::env::var("XLA_FLAGS").is_err() {
        use std::{os::unix::process::CommandExt, process::Command};

        let curr_exe = std::env::current_exe().unwrap();
        let runner_dir = curr_exe.parent().unwrap();
        let curr_path = std::env::var("PATH").unwrap();
        let ld_library_path = std::env::var("LD_LIBRARY_PATH").unwrap_or_default();

        let mut args = std::env::args().into_iter();
        let arg0 = args.next().unwrap();

        // reexec the current process with the correct args
        Command::new("/proc/self/exe")
            .arg0(arg0)
            .args(args)
            .env(
                "XLA_FLAGS",
                format!(
                    "--xla_gpu_cuda_data_dir={}",
                    runner_dir.join("nvcc/nvidia/cuda_nvcc").to_str().unwrap()
                ),
            )
            .env(
                "PATH",
                format!(
                    "${curr_path}:{}",
                    runner_dir
                        .join("nvcc/nvidia/cuda_nvcc/bin")
                        .to_str()
                        .unwrap()
                ),
            )
            .env(
                "LD_LIBRARY_PATH",
                format!(
                    "${ld_library_path}:{}:{}:{}",
                    runner_dir
                        .join("cudart/nvidia/cuda_runtime/lib")
                        .to_str()
                        .unwrap(),
                    runner_dir.join("cudnn/nvidia/cudnn/lib").to_str().unwrap(),
                    runner_dir
                        .join("cublas/nvidia/cublas/lib")
                        .to_str()
                        .unwrap(),
                ),
            )
            .exec();
    }

    let mut server = init_runner().await;
    let mut model = None;

    let mut token_gen = 0;
    let mut sealed_tensors = HashMap::new();

    while let Some(req) = server.get_next_request().await {
        let req_id = req.id;
        match req.data {
            RequestData::Load { fs, .. } => {
                let fs = server.get_readonly_filesystem(fs).await.unwrap();
                let data = fs.read("model.pb").await.unwrap();

                // Try to load on GPU if we can
                let client = xla::PjRtClient::gpu(1.0, false)
                    .unwrap_or_else(|_| xla::PjRtClient::cpu().unwrap());
                let proto = xla::HloModuleProto::parse_proto(&data, true).unwrap();
                let comp = xla::XlaComputation::from_proto(&proto);

                // Load the model info
                let info = fs.read("model.json").await.unwrap();
                let info: ModelInfo = serde_json::from_slice(&info).unwrap();

                model = Some(Model {
                    executable: client.compile(&comp).unwrap(),
                    info,
                });

                server
                    .send_response_for_request(req_id, ResponseData::Load)
                    .await
                    .unwrap();
            }

            RequestData::Pack { fs, input_path, .. } => {
                // Verify that we have info in the expected format
                let fs = server.get_readonly_filesystem(fs).await.unwrap();
                let input_path = PathBuf::from(input_path);

                // TODO: don't unwrap
                let info = fs.read(input_path.join("model.json")).await.unwrap();
                let _: ModelInfo = serde_json::from_slice(&info).unwrap();

                // TODO Return an error instead of asserting
                assert!(fs.metadata(input_path.join("model.pb")).await.is_ok());

                // Just return the input path
                server
                    .send_response_for_request(
                        req_id,
                        ResponseData::Pack {
                            output_path: input_path.to_string(),
                        },
                    )
                    .await
                    .unwrap();
            }

            RequestData::Seal { tensors } => {
                // Generate a token and store the tensors
                let handle = SealHandle::new(token_gen);
                sealed_tensors.insert(handle, tensors);
                token_gen += 1;
                server
                    .send_response_for_request(req_id, ResponseData::Seal { handle })
                    .await
                    .unwrap();
            }

            RequestData::InferWithTensors { tensors, .. } => {
                // TODO: infer on another thread or use spawn blocking
                let output = model.as_ref().unwrap().infer(tensors);
                // Let's just return the input tensors for now
                server
                    .send_response_for_request(req_id, ResponseData::Infer { tensors: output })
                    .await
                    .unwrap();
            }

            RequestData::InferWithHandle { handle, .. } => {
                // TODO: return an error instead of using unwrap
                let tensors = sealed_tensors.remove(&handle).unwrap();

                // TODO: infer on another thread or use spawn blocking
                let output = model.as_ref().unwrap().infer(tensors);

                // Let's just return the input tensors for now
                server
                    .send_response_for_request(req_id, ResponseData::Infer { tensors: output })
                    .await
                    .unwrap();
            }
        }
    }
}

#[derive(Serialize, Deserialize)]
struct ModelInfo {
    input_ordering: Vec<String>,
    output_ordering: Vec<String>,
}

struct Model {
    executable: PjRtLoadedExecutable,
    info: ModelInfo,
}

impl Model {
    fn infer(&self, mut tensors: HashMap<String, Tensor>) -> HashMap<String, Tensor> {
        // Convert the input types
        let input_tensors: Vec<_> = self
            .info
            .input_ordering
            .iter()
            .map(|k| tensor_to_literal(tensors.remove(k).unwrap()))
            .collect();

        // Run inference
        let mut out = self.executable.execute(&input_tensors).unwrap();

        // `out` should be a single output that's a tuple
        assert_eq!(out.len(), 1);

        // TODO: figure out what replicas are in XLA
        let mut out = out.remove(0);
        assert_eq!(out.len(), 1);

        // TODO: maybe async copy to CPU in the future
        let tuple = out.remove(0).to_literal_sync().unwrap();

        // Convert the literal to a tuple
        let tuple = tuple.to_tuple().unwrap();

        // Convert output types
        tuple
            .into_iter()
            .zip(self.info.output_ordering.iter())
            .map(|(v, key)| (key.clone(), literal_to_tensor(v)))
            .collect()
    }
}
