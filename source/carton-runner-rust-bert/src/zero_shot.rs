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

use async_trait::async_trait;
use carton_runner_interface::types::{Tensor, TensorStorage};
use lunchbox::{types::ReadableFile, ReadableFileSystem};
use rust_bert::{
    pipelines::{
        common::{ModelResource, ModelType},
        zero_shot_classification::{ZeroShotClassificationConfig, ZeroShotClassificationModel},
    },
    resources::LocalResource,
};
use serde::{Deserialize, Serialize};

use crate::{copy_to_local, Model, ModelFromConfig};

/// Config for a zero shot classification model
#[derive(Serialize, Deserialize)]
pub struct CartonZeroShotConfig {
    model_type: ModelType,
    model_path: String,
    config_path: String,
    vocab_path: String,
    merges_path: Option<String>,
}

pub struct CartonZeroShotModel {
    _tempdir: tempfile::TempDir,
    model: ZeroShotClassificationModel,
}

#[async_trait]
impl ModelFromConfig for CartonZeroShotConfig {
    type ModelType = CartonZeroShotModel;

    async fn load<F>(self, fs: &F) -> Self::ModelType
    where
        F: ReadableFileSystem + Send + Sync,
        F::FileType: ReadableFile + Unpin + Send + Sync,
    {
        let td = tempfile::tempdir().unwrap();
        let base = td.path();
        // Load all the model resources
        tokio::join!(
            copy_to_local(fs, base, &self.model_path),
            copy_to_local(fs, base, &self.config_path),
            copy_to_local(fs, base, &self.vocab_path),
            async {
                if let Some(p) = &self.merges_path {
                    copy_to_local(fs, base, p).await;
                }
            },
        );

        log::trace!("Loading zero shot classification model...");
        // Defaults to cuda if available
        let zero_shot_config = ZeroShotClassificationConfig::new(
            self.model_type,
            ModelResource::Torch(td.path().join(self.model_path).into()),
            LocalResource::from(td.path().join(self.config_path)),
            LocalResource::from(td.path().join(self.vocab_path)),
            self.merges_path
                .map(|p| LocalResource::from(td.path().join(p))),
            false,
            None,
            None,
        );

        let model = ZeroShotClassificationModel::new(zero_shot_config).unwrap();

        CartonZeroShotModel {
            _tempdir: td,
            model,
        }
    }
}

impl Model for CartonZeroShotModel {
    fn infer(&self, tensors: HashMap<String, Tensor>) -> HashMap<String, Tensor> {
        // TODO: don't unwrap
        let input_tensor = tensors.get("input").unwrap();
        let candidate_labels = tensors.get("candidate_labels").unwrap();
        let template = tensors.get("template");
        let max_length = tensors.get("max_length");

        // Get all of them as string tensors
        if let Tensor::String(input_tensor) = input_tensor {
            let input_tensor = input_tensor.view();

            if let Tensor::String(candidate_labels) = candidate_labels {
                let candidate_labels = candidate_labels.view();

                // Create an output tensor with the appropriate shape (input shape with an extra dimension)
                let mut output_tensor = TensorStorage::new(
                    input_tensor
                        .shape()
                        .iter()
                        .chain(&[candidate_labels.len()])
                        .map(|v| (*v) as _)
                        .collect(),
                );

                // Reshape to [input_tensor.len(), candidate_labels.len()]
                let mut output_view = output_tensor
                    .view_mut()
                    .into_shape([input_tensor.len(), candidate_labels.len()])
                    .unwrap();

                // Fill with zeros
                output_view.fill(0f32);

                let template = template.map(|t| {
                    if let Tensor::String(t) = t {
                        let format_str = t.view().first().unwrap().to_owned();

                        // We can't use dynamic format strings so lets just replace {} for now
                        // TODO: improve
                        Box::new(move |label: &str| format_str.replace("{}", label)) as _
                    } else {
                        // TODO: don't do this
                        panic!("Tensor `template` exists, but did not contain strings")
                    }
                });

                let max_length = max_length.map_or(128, |t| {
                    if let Tensor::U32(t) = t {
                        t.view().first().unwrap().to_owned()
                    } else {
                        panic!("Tensor `max_length` exists, but did not contain u32s")
                    }
                });

                let predicted = self
                    .model
                    .predict_multilabel(
                        input_tensor
                            .as_slice()
                            .unwrap()
                            .into_iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>(),
                        candidate_labels
                            .as_slice()
                            .unwrap()
                            .into_iter()
                            .map(|s| s.as_str())
                            .collect::<Vec<_>>(),
                        template,
                        max_length as _,
                    )
                    .unwrap();

                for (i, labels) in predicted.into_iter().enumerate() {
                    // Set the values of the output tensor
                    let mut indexed_output_view = output_view.index_axis_mut(ndarray::Axis(0), i);
                    let sliced_output_view = indexed_output_view.as_slice_mut().unwrap();

                    for label in labels {
                        sliced_output_view[label.id as usize] = label.score as _;
                    }
                }

                let mut out = HashMap::new();
                out.insert("scores".to_owned(), Tensor::Float(output_tensor));
                return out;
            }
        }

        // TODO: don't do this
        panic!("Unexpected input");
    }
}

pub mod pack {
    use std::path::PathBuf;

    use carton::{
        info::{
            CartonInfo, DataType, Dimension, Example, LinkedFile, RunnerInfo, Shape, TensorOrMisc,
            TensorSpec,
        },
        types::{PackOpts, Tensor},
    };

    use crate::{download_file, ModelConfig};

    pub async fn pack_bart_mnli() -> PathBuf {
        let model_config = ModelConfig::ZeroShotClassification(super::CartonZeroShotConfig {
            model_type: rust_bert::pipelines::common::ModelType::Bart,
            model_path: "./model/rust_model.ot".into(),
            config_path: "./model/config.json".into(),
            vocab_path: "./model/vocab.json".into(),
            merges_path: Some("./model/merges.txt".into()),
        });

        // Create a tempdir to pack
        let dir = tempfile::tempdir().unwrap();

        // Write the config
        let serialized = serde_json::to_vec(&model_config).unwrap();
        tokio::fs::write(dir.path().join("config.json"), serialized)
            .await
            .unwrap();

        // Add the model resources
        let model_dir = dir.path().join("model");
        tokio::fs::create_dir(&model_dir).await.unwrap();
        let res = tokio::join!(
            download_file(
                LinkedFile {
                    urls: vec!["https://huggingface.co/facebook/bart-large-mnli/resolve/9fc9c4e1808b5613968646fa771fc43fb03995f2/rust_model.ot".into()],
                    sha256: "b48c2b60d9a63b6ad67d99720b4d41ecb235287f10fcaeaae412291cdaf28578".into(),
                },
                model_dir.join("rust_model.ot"),
            ),
            download_file(
                LinkedFile {
                    urls: vec!["https://huggingface.co/facebook/bart-large-mnli/resolve/9fc9c4e1808b5613968646fa771fc43fb03995f2/config.json".into()],
                    sha256: "a0f9bcb245b680a96ccae0ad8d155f267ec3e3c971ef4a4937e52ea9ba368a86".into(),
                },
                model_dir.join("config.json"),
            ),
            download_file(
                LinkedFile {
                    urls: vec!["https://huggingface.co/facebook/bart-large-mnli/resolve/9fc9c4e1808b5613968646fa771fc43fb03995f2/vocab.json".into()],
                    sha256: "06b4d46c8e752d410213d9548eb27a54db70fda0319b6271fb8d59dead5e1cab".into(),
                },
                model_dir.join("vocab.json"),
            ),
            download_file(
                LinkedFile {
                    urls: vec!["https://huggingface.co/facebook/bart-large-mnli/resolve/9fc9c4e1808b5613968646fa771fc43fb03995f2/merges.txt".into()],
                    sha256: "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5".into(),
                },
                model_dir.join("merges.txt"),
            ),
        );

        // TODO: better error handling
        let linked_files = vec![
            res.0.unwrap(),
            res.1.unwrap(),
            res.2.unwrap(),
            res.3.unwrap(),
        ];

        // Pack the model and return the path
        let info = CartonInfo {
            model_name: Some("BART Large MNLI".into()),
            short_description: Some("BART Large MNLI is a model that can do zero shot classificiation.".into()),
            model_description: Some("See [here](https://huggingface.co/facebook/bart-large-mnli) for more details.\n\nNote: This model performs multi-label classification (i.e. zero or more labels may be true for each input).".into()),
            license: Some("MIT".into()),
            repository: None,
            homepage: Some("https://huggingface.co/facebook/bart-large-mnli".into()),
            required_platforms: None,
            inputs: Some(vec![
                TensorSpec {
                    name: "input".into(),
                    dtype: DataType::String,
                    shape: Shape::Shape(vec![Dimension::Symbol("N".into())]),
                    description: Some("The strings to classifiy".into()),
                    internal_name: None
                },
                TensorSpec {
                    name: "candidate_labels".into(),
                    dtype: DataType::String,
                    shape: Shape::Shape(vec![Dimension::Symbol("L".into())]),
                    description: Some("The candidate labels".into()),
                    internal_name: None
                },
                TensorSpec {
                    name: "template".into(),
                    dtype: DataType::String,
                    shape: Shape::Shape(vec![]),
                    description: Some("An optional template string for the model to use. Defaults to 'This example is about {}.'".into()),
                    internal_name: None
                },
                TensorSpec {
                    name: "max_length".into(),
                    dtype: DataType::U32,
                    shape: Shape::Shape(vec![]),
                    description: Some("An optional max_length to pass to the model. Defaults to 128.".into()),
                    internal_name: None
                }
            ]),
            outputs: Some(vec![
                TensorSpec {
                    name: "scores".into(),
                    dtype: DataType::String,
                    shape: Shape::Shape(vec![Dimension::Symbol("N".into()), Dimension::Symbol("L".into())]),
                    description: Some("Scores between 0 and 1 for each element of `input` for each label in `candidate_labels`".into()),
                    internal_name: None
                },
            ]),
            self_tests: None,
            examples: Some(vec![
                Example {
                    name: Some("quickstart".into()),
                    description: Some("".into()),
                    inputs: [
                        ("input".into(), TensorOrMisc::Tensor(Tensor::String(ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec!["The Dow Jones Industrial Average is a stock market index of 30 prominent companies listed on stock exchanges in the United States.".into(), "The prime minister has announced a stimulus package which was widely criticized by the opposition".into()]).unwrap().into()).into())),
                        ("candidate_labels".into(), TensorOrMisc::Tensor(Tensor::String(ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[4]), vec!["politics".into(), "public health".into(), "economics".into(), "sports".into()]).unwrap().into()).into())),
                        ("max_length".into(), TensorOrMisc::Tensor(Tensor::U32(ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[]), vec![256]).unwrap().into()).into())),
                    ].into(),
                    sample_out: [
                        ("scores".into(), TensorOrMisc::Tensor(Tensor::Float(ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 4]), vec![0.00058022776, 0.00047010265, 0.035326574, 0.00057026354, 0.9282547, 0.0029879552, 0.8838335, 0.0003471978]).unwrap().into()).into()))
                    ].into(),
                }
            ]),
            runner: RunnerInfo {
                runner_name: "rust-bert".into(),
                required_framework_version: semver::VersionReq::parse(">= 0.0.0").unwrap(),
                runner_compat_version: None,
                opts: None,
            },
            misc_files: None,
        };

        carton::Carton::pack(
            dir.path().to_str().unwrap().to_owned(),
            PackOpts {
                info,
                linked_files: Some(linked_files),
            },
        )
        .await
        .unwrap()
    }
}
