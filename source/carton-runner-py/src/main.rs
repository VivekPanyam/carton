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
    server::{init_runner, RequestData, ResponseData, Server},
    types::Tensor,
};

use futures_util::{pin_mut, StreamExt};
use packager::update_or_generate_lockfile;

mod env;
mod loader;
mod model;
mod packager;
mod pip_utils;
mod python_utils;
mod wheel;

// This is basically the expanded version of
// #[pyo3_asyncio::tokio::main]
// but modified to call `crate::python_utils::init();` before setting up python
fn main() {
    async fn main() -> pyo3::PyResult<()> {
        main_inner().await;
        Ok(())
    }

    // Setup the isolated python env
    crate::python_utils::init();

    pyo3::prepare_freethreaded_python();
    let mut builder = pyo3_asyncio::tokio::re_exports::runtime::Builder::new_multi_thread();
    builder.enable_all();
    pyo3_asyncio::tokio::init(builder);
    pyo3::Python::with_gil(|py| {
        pyo3_asyncio::tokio::run(py, main())
            .map_err(|e| {
                e.print_and_set_sys_last_vars(py);
            })
            .unwrap();
    });
}

async fn main_inner() {
    let mut server = init_runner().await;

    let mut model = None;

    while let Some(req) = server.get_next_request().await {
        let req_id = req.id;
        match req.data {
            RequestData::Load {
                fs, runner_opts, ..
            } => match crate::loader::load(
                server.get_readonly_filesystem(fs).await.unwrap(),
                runner_opts,
            )
            .await
            {
                Ok(m) => {
                    // Store the model
                    model = Some(m);

                    // Send a response
                    server
                        .send_response_for_request(req_id, ResponseData::Load)
                        .await
                        .unwrap()
                }
                Err(e) => server
                    .send_response_for_request(req_id, ResponseData::Error { e })
                    .await
                    .unwrap(),
            },
            RequestData::Pack { fs, input_path, .. } => {
                let fs = server.get_writable_filesystem(fs).await.unwrap();

                // Update or generate a lockfile in the input dir
                update_or_generate_lockfile(&fs, &input_path).await;

                // The dir that carton should pack is just the input path
                server
                    .send_response_for_request(
                        req_id,
                        ResponseData::Pack {
                            output_path: input_path,
                        },
                    )
                    .await
                    .unwrap();
            }
            RequestData::Seal { tensors } => {
                // Call `model.seal`
                match model.as_mut().unwrap().seal(tensors) {
                    Ok(handle) => server
                        .send_response_for_request(req_id, ResponseData::Seal { handle })
                        .await
                        .unwrap(),
                    Err(e) => server
                        .send_response_for_request(
                            req_id,
                            ResponseData::Error {
                                e: format!("Error calling `seal` method on model: {e}"),
                            },
                        )
                        .await
                        .unwrap(),
                }
            }
            RequestData::InferWithTensors { tensors, streaming } => {
                // Call `model.infer_with_tensors`
                let res = model.as_mut().unwrap().infer_with_tensors(tensors).await;
                send_infer_response(&server, res, streaming, req_id, "infer_with_tensors").await;
            }
            RequestData::InferWithHandle { handle, streaming } => {
                // Call `model.infer_with_handle`
                let res = model.as_mut().unwrap().infer_with_handle(handle).await;
                send_infer_response(&server, res, streaming, req_id, "infer_with_handle").await;
            }
        }
    }
}

fn transform_res(v: Result<HashMap<String, Tensor>, String>, method: &'static str) -> ResponseData {
    match v {
        Ok(out) => ResponseData::Infer { tensors: out },
        Err(e) => ResponseData::Error {
            e: format!("Error calling `{method}` method on model: {e}"),
        },
    }
}

/// A utility to send inference responses
async fn send_infer_response(
    server: &Server,
    res: Result<impl futures::Stream<Item = Result<HashMap<String, Tensor>, String>>, String>,
    streaming: bool,
    req_id: u64,
    method: &'static str,
) {
    match res {
        Ok(stream) => {
            pin_mut!(stream);

            let mut last_val = None;
            while let Some(item) = stream.next().await {
                if streaming {
                    server
                        .send_streaming_response_for_request(
                            req_id,
                            false,
                            transform_res(item, method),
                        )
                        .await
                        .unwrap()
                } else {
                    // Not a streaming response so just store the values
                    last_val = Some(item);
                }
            }

            if streaming {
                // If we're sending a streaming response, send a completion message
                server
                    .send_streaming_response_for_request(req_id, true, ResponseData::Empty)
                    .await
                    .unwrap()
            } else {
                server
                    .send_response_for_request(req_id, transform_res(last_val.unwrap(), method))
                    .await
                    .unwrap()
            }
        }
        Err(e) => server
            .send_response_for_request(
                req_id,
                ResponseData::Error {
                    e: format!("Error calling `{method}` method on model: {e}"),
                },
            )
            .await
            .unwrap(),
    }
}
