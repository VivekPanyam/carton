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
        masked_language::{MaskedLanguageConfig, MaskedLanguageModel},
    },
    resources::LocalResource,
};
use serde::{Deserialize, Serialize};

use crate::{copy_to_local, Model, ModelFromConfig};

/// Config for a masked language model
#[derive(Serialize, Deserialize)]
pub struct CartonMaskedLanguageConfig {
    model_type: ModelType,
    model_path: String,
    config_path: String,
    vocab_path: String,
    merges_path: Option<String>,
    lower_case: bool,
    strip_accents: Option<bool>,
    add_prefix_space: Option<bool>,
}

pub struct CartonMaskedLanguageModel {
    _tempdir: tempfile::TempDir,
    model: MaskedLanguageModel,
}

#[async_trait]
impl ModelFromConfig for CartonMaskedLanguageConfig {
    type ModelType = CartonMaskedLanguageModel;

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

        log::trace!("Loading masked language model...");
        // Defaults to cuda if available
        let masked_language_config = MaskedLanguageConfig::new(
            self.model_type,
            ModelResource::Torch(td.path().join(self.model_path).into()),
            LocalResource::from(td.path().join(self.config_path)),
            LocalResource::from(td.path().join(self.vocab_path)),
            self.merges_path
                .map(|p| LocalResource::from(td.path().join(p))),
            self.lower_case,
            self.strip_accents,
            self.add_prefix_space,
            Some(String::from("[MASK]")),
        );

        let model = MaskedLanguageModel::new(masked_language_config).unwrap();

        CartonMaskedLanguageModel {
            _tempdir: td,
            model,
        }
    }
}

impl Model for CartonMaskedLanguageModel {
    fn infer(&self, tensors: HashMap<String, Tensor>) -> HashMap<String, Tensor> {
        // TODO: don't unwrap
        let input_tensor = tensors.get("input").unwrap();
        let max_tokens = tensors.get("max_tokens").map_or(1, |t| {
            // TODO: handle other integer types
            if let Tensor::U32(t) = t {
                t.view().first().unwrap().to_owned()
            } else {
                panic!("Tensor `max_tokens` exists, but did not contain u32s")
            }
        });

        if let Tensor::String(input_tensor) = input_tensor {
            let input_tensor = input_tensor.view();

            // Create an output token tensor with shape [input_shape, max_tokens]
            let mut tokens_output_tensor = TensorStorage::new(
                input_tensor
                    .shape()
                    .iter()
                    .map(|v| (*v) as _)
                    .chain([max_tokens as _])
                    .collect(),
            );
            let mut tokens_output_view = tokens_output_tensor.view_mut();

            // Create an output scores tensor with shape [input_shape, max_tokens]
            let mut scores_output_tensor = TensorStorage::new(
                input_tensor
                    .shape()
                    .iter()
                    .map(|v| (*v) as _)
                    .chain([max_tokens as _])
                    .collect(),
            );
            let mut scores_output_view = scores_output_tensor.view_mut();

            // Come up with candidate tokens and store in the output
            let masked_values = self
                .model
                .predict(
                    input_tensor
                        .as_slice()
                        .unwrap()
                        .into_iter()
                        .map(|v| v.as_str())
                        .collect::<Vec<_>>(),
                )
                .unwrap();

            for (i, mut item) in masked_values.into_iter().enumerate() {
                let mut indexed_tokens_output_view =
                    tokens_output_view.index_axis_mut(ndarray::Axis(0), i);
                let sliced_tokens_output_view = indexed_tokens_output_view.as_slice_mut().unwrap();

                let mut indexed_scores_output_view =
                    scores_output_view.index_axis_mut(ndarray::Axis(0), i);
                let sliced_scores_output_view = indexed_scores_output_view.as_slice_mut().unwrap();

                // Descending sort
                item.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

                // Zip will keep going until the shortest sequence runs out. Since this is limited to `max_tokens`, we don't
                // need to specifically handle it.
                for ((token, score), val) in std::iter::zip(
                    std::iter::zip(sliced_tokens_output_view, sliced_scores_output_view),
                    item,
                ) {
                    *token = val.text;
                    *score = val.score as f32;
                }
            }

            let mut out = HashMap::new();
            out.insert("tokens".to_owned(), Tensor::String(tokens_output_tensor));
            out.insert("scores".to_owned(), Tensor::Float(scores_output_tensor));
            return out;
        }

        // TODO: don't do this
        panic!("Unexpected input");
    }
}

pub mod pack {
    use std::path::PathBuf;

    use carton::{
        info::{
            DataType, Dimension, Example, LinkedFile, RunnerInfo, Shape, TensorOrMisc, TensorSpec,
        },
        types::{CartonInfo, PackOpts, Tensor},
    };

    use crate::{download_file, ModelConfig};

    pub async fn pack_bert_base_uncased() -> PathBuf {
        let model_config = ModelConfig::FillMask(super::CartonMaskedLanguageConfig {
            model_type: rust_bert::pipelines::common::ModelType::Bert,
            model_path: "./model/rust_model.ot".into(),
            config_path: "./model/config.json".into(),
            vocab_path: "./model/vocab.txt".into(),
            merges_path: None,
            lower_case: true,
            strip_accents: None,
            add_prefix_space: None,
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
                    urls: vec!["https://huggingface.co/bert-base-uncased/resolve/1dbc166cf8765166998eff31ade2eb64c8a40076/rust_model.ot".into()],
                    sha256: "afd9aa425fd45c5655d3d43a0d041f9b76729bf475d6c017a0e9304a38f89972".into(),
                },
                model_dir.join("rust_model.ot"),
            ),
            download_file(
                LinkedFile {
                    urls: vec!["https://huggingface.co/bert-base-uncased/resolve/1dbc166cf8765166998eff31ade2eb64c8a40076/config.json".into()],
                    sha256: "7160e1553ad2ca51d8c1cb066be533db31826e12d173824c1bb0cb1a4f187d20".into(),
                },
                model_dir.join("config.json"),
            ),
            download_file(
                LinkedFile {
                    urls: vec!["https://huggingface.co/bert-base-uncased/resolve/1dbc166cf8765166998eff31ade2eb64c8a40076/vocab.txt".into()],
                    sha256: "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3".into(),
                },
                model_dir.join("vocab.txt"),
            ),
        );

        // TODO: better error handling
        let linked_files = vec![res.0.unwrap(), res.1.unwrap(), res.2.unwrap()];

        // Pack the model and return the path
        let info = CartonInfo {
            model_name: Some("bert-base-uncased".into()),
            short_description: Some("A language model that can fill masked tokens in sentences".into()),
            model_description: Some("This model can predict masked tokens in a sentence. For example, it might predict `capital` given `Paris is the [MASK] of France.`.\n\nSee [here](https://huggingface.co/bert-base-uncased) for more details.".into()),
            license: Some("Apache-2.0".into()),
            repository: None,
            homepage: Some("https://github.com/google-research/bert".into()),
            required_platforms: None,
            inputs: Some(vec![
                TensorSpec {
                    name: "input".into(),
                    dtype: DataType::String,
                    shape: Shape::Shape(vec![Dimension::Symbol("N".into())]),
                    description: Some("The sentences to fill `[MASK]` tokens in.".into()),
                    internal_name: None
                },
                TensorSpec {
                    name: "max_tokens".into(),
                    dtype: DataType::U32,
                    shape: Shape::Shape(vec![]),
                    description: Some("The maximum number of tokens to predict for each mask. Optional, defaults to 1.".into()),
                    internal_name: None
                },
            ]),
            outputs: Some(vec![
                TensorSpec {
                    name: "tokens".into(),
                    dtype: DataType::String,
                    shape: Shape::Shape(vec![Dimension::Symbol("N".into()), Dimension::Any]),
                    description: Some("The predicted tokens for each input sentence. This will have shape `[N, max_tokens]`, but some cells may be empty.".into()),
                    internal_name: None
                },
                TensorSpec {
                    name: "scores".into(),
                    dtype: DataType::Float,
                    shape: Shape::Shape(vec![Dimension::Symbol("N".into()), Dimension::Any]),
                    description: Some("The scores for each predicted token. This will have shape `[N, max_tokens]`, but some cells may have a score of zero.".into()),
                    internal_name: None
                },
            ]),
            self_tests: None,
            examples: Some(vec![
                Example {
                    name: Some("quickstart".into()),
                    description: None,
                    inputs: [
                        ("input".into(), TensorOrMisc::Tensor(Tensor::String(ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec!["Paris is the [MASK] of France.".into(), "Today is a good [MASK].".into()]).unwrap().into()).into())),
                    ].into(),
                    sample_out: [
                        ("tokens".into(), TensorOrMisc::Tensor(Tensor::String(ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 1]), vec!["capital".into(), "day".into()]).unwrap().into()).into())),
                        ("scores".into(), TensorOrMisc::Tensor(Tensor::Float(ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 1]), vec![18.19973, 12.977381]).unwrap().into()).into()))
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
