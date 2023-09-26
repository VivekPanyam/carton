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
        sentiment::{SentimentModel, SentimentPolarity},
        sequence_classification::SequenceClassificationConfig,
    },
    resources::LocalResource,
};
use serde::{Deserialize, Serialize};

use crate::{copy_to_local, Model, ModelFromConfig};

/// Config for a sentiment analysis model
#[derive(Serialize, Deserialize)]
pub struct CartonSentimentAnalysisConfig {
    model_type: ModelType,
    model_path: String,
    config_path: String,
    vocab_path: String,
    merges_path: Option<String>,
}

pub struct CartonSentimentAnalysisModel {
    _tempdir: tempfile::TempDir,
    model: SentimentModel,
}

#[async_trait]
impl ModelFromConfig for CartonSentimentAnalysisConfig {
    type ModelType = CartonSentimentAnalysisModel;

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

        log::trace!("Loading sentiment analysis model...");
        // Defaults to cuda if available
        let sentiment_config = SequenceClassificationConfig::new(
            self.model_type,
            ModelResource::Torch(td.path().join(self.model_path).into()),
            LocalResource::from(td.path().join(self.config_path)),
            LocalResource::from(td.path().join(self.vocab_path)),
            self.merges_path
                .map(|p| LocalResource::from(td.path().join(p))),
            true,
            None,
            None,
        );

        let model = SentimentModel::new(sentiment_config).unwrap();

        CartonSentimentAnalysisModel {
            _tempdir: td,
            model,
        }
    }
}

impl Model for CartonSentimentAnalysisModel {
    fn infer(&self, tensors: HashMap<String, Tensor>) -> HashMap<String, Tensor> {
        // TODO: don't unwrap
        let input_tensor = tensors.get("input").unwrap();

        // Get all of them as string tensors
        if let Tensor::String(input_tensor) = input_tensor {
            let input_tensor = input_tensor.view();

            // Create an output tensor with the same shape
            let mut output_tensor =
                TensorStorage::new(input_tensor.shape().iter().map(|v| (*v) as _).collect());
            let mut output_view = output_tensor.view_mut();
            let sliced_output_view = output_view.as_slice_mut().unwrap();

            let predictions = self.model.predict(
                input_tensor
                    .as_slice()
                    .unwrap()
                    .into_iter()
                    .map(|item| item.as_str())
                    .collect::<Vec<_>>(),
            );

            for (sentiment, out) in predictions.into_iter().zip(sliced_output_view) {
                match sentiment.polarity {
                    SentimentPolarity::Positive => *out = sentiment.score as f32,
                    SentimentPolarity::Negative => *out = -1f32 * (sentiment.score as f32),
                }
            }

            let mut out = HashMap::new();
            out.insert("scores".to_owned(), Tensor::Float(output_tensor));
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
            CartonInfo, DataType, Example, LinkedFile, RunnerInfo, Shape, TensorOrMisc, TensorSpec,
        },
        types::{GenericStorage, PackOpts, Tensor},
    };

    use crate::{download_file, ModelConfig};

    pub async fn pack_distilbert_sst2() -> PathBuf {
        let model_config = ModelConfig::SentimentAnalysis(super::CartonSentimentAnalysisConfig {
            model_type: rust_bert::pipelines::common::ModelType::DistilBert,
            model_path: "./model/rust_model.ot".into(),
            config_path: "./model/config.json".into(),
            vocab_path: "./model/vocab.txt".into(),
            merges_path: None,
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
                    urls: vec!["https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/resolve/3d65bad49c7ba6f71920504507a8927f4b9db6c0/rust_model.ot".into()],
                    sha256: "9db97da21b97a5e6db1212ce6a810a0c5e22c99daefe3355bae2117f78a0abb9".into(),
                },
                model_dir.join("rust_model.ot"),
            ),
            download_file(
                LinkedFile {
                    urls: vec!["https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/resolve/3d65bad49c7ba6f71920504507a8927f4b9db6c0/config.json".into()],
                    sha256: "582122c8f414793d131e10022ce9ba04e3811a9da6389137ee2f18665b4f4d15".into(),
                },
                model_dir.join("config.json"),
            ),
            download_file(
                LinkedFile {
                    urls: vec!["https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english/resolve/3d65bad49c7ba6f71920504507a8927f4b9db6c0/vocab.txt".into()],
                    sha256: "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3".into(),
                },
                model_dir.join("vocab.txt"),
            ),
        );

        // TODO: better error handling
        let linked_files = vec![res.0.unwrap(), res.1.unwrap(), res.2.unwrap()];

        // Pack the model and return the path
        let info = CartonInfo::<GenericStorage> {
            model_name: Some("DistilBERT base uncased finetuned SST-2".into()),
            short_description: Some("DistilBERT base uncased finetuned SST-2 is a model that can do sentiment analysis.".into()),
            model_description: Some("See [here](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) for more details.".into()),
            license: Some("Apache-2.0".into()),
            repository: None,
            homepage: Some("https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english".into()),
            required_platforms: None,
            inputs: Some(vec![
                TensorSpec {
                    name: "input".into(),
                    dtype: DataType::String,
                    shape: Shape::Symbol("input_shape".into()),
                    description: Some("The strings to analyze the sentiment of".into()),
                    internal_name: None
                },
            ]),
            outputs: Some(vec![
                TensorSpec {
                    name: "scores".into(),
                    dtype: DataType::String,
                    shape: Shape::Symbol("input_shape".into()),
                    description: Some("Scores between -1 and 1 for each element of `input`. Negative scores correspond to a negative sentiment.".into()),
                    internal_name: None
                },
            ]),
            self_tests: None,
            examples: Some(vec![
                Example {
                    name: Some("quickstart".into()),
                    description: Some("".into()),
                    inputs: [
                        ("input".into(), TensorOrMisc::Tensor(Tensor::String(ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec!["I love pizza".into(), "This car is fast, but gets hot.".into(), "Most movies that try to do too many things are bad, but this one was different.".into()]).unwrap().into()).into())),
                    ].into(),
                    sample_out: [
                        ("scores".into(), TensorOrMisc::Tensor(Tensor::Float(ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec![0.97580576, -0.74823254, 0.729913]).unwrap()).into()))
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
