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

use carton_runner_interface::server::{init_runner, RequestData, ResponseData, SealHandle};
use carton_runner_rust_bert::{Model, ModelConfig, ModelFromConfig};
use lunchbox::ReadableFileSystem;

#[tokio::main]
async fn main() {
    let mut server = init_runner().await;

    let mut sealed = HashMap::new();
    let mut seal_counter = 0;

    let mut model: Option<Box<dyn Model>> = None;

    while let Some(req) = server.get_next_request().await {
        let req_id = req.id;
        match req.data {
            RequestData::Load {
                fs, runner_opts: _, ..
            } => {
                // Load the model config
                let fs = server.get_readonly_filesystem(fs).await.unwrap();
                let config: ModelConfig =
                    serde_json::from_slice(&fs.read("config.json").await.unwrap()).unwrap();

                match config {
                    ModelConfig::Translation(config) => {
                        model = Some(Box::new(config.load(&fs).await))
                    }
                    ModelConfig::Summarization(config) => {
                        model = Some(Box::new(config.load(&fs).await))
                    }
                    ModelConfig::ZeroShotClassification(config) => {
                        model = Some(Box::new(config.load(&fs).await))
                    }
                    ModelConfig::SentimentAnalysis(config) => {
                        model = Some(Box::new(config.load(&fs).await))
                    }
                    ModelConfig::NER => todo!(),
                    ModelConfig::POSTagging => todo!(),
                    ModelConfig::QuestionAnswering(config) => {
                        model = Some(Box::new(config.load(&fs).await))
                    }
                    ModelConfig::KeywordExtraction => todo!(),
                    ModelConfig::TextClassification => todo!(),
                    ModelConfig::FillMask(config) => model = Some(Box::new(config.load(&fs).await)),
                    ModelConfig::SentenceEmbeddings => todo!(),
                    ModelConfig::TextGeneration(config) => {
                        model = Some(Box::new(config.load(&fs).await))
                    }
                }

                server
                    .send_response_for_request(req_id, ResponseData::Load)
                    .await
                    .unwrap();
            }
            RequestData::Pack { input_path, .. } => {
                // This should basically be a noop since the structure of the input folder should be the same as the target
                // Just return the input path
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
                sealed.insert(seal_counter, tensors);

                server
                    .send_response_for_request(
                        req_id,
                        ResponseData::Seal {
                            handle: SealHandle::new(seal_counter),
                        },
                    )
                    .await
                    .unwrap();

                seal_counter += 1;
            }
            RequestData::InferWithTensors { tensors, .. } => {
                // TODO: error handling
                let result = model.as_ref().map(|m| m.infer(tensors));

                server
                    .send_response_for_request(
                        req_id,
                        ResponseData::Infer {
                            tensors: result.unwrap(),
                        },
                    )
                    .await
                    .unwrap();
            }
            RequestData::InferWithHandle { handle, .. } => {
                // TODO: error handling
                let result = sealed
                    .remove(&handle.get())
                    .and_then(|tensors| model.as_ref().map(|m| m.infer(tensors)));

                server
                    .send_response_for_request(
                        req_id,
                        ResponseData::Infer {
                            tensors: result.unwrap(),
                        },
                    )
                    .await
                    .unwrap();
            }
        }
    }
}
