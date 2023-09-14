use std::collections::HashMap;

use async_trait::async_trait;
use carton_runner_interface::types::{Tensor, TensorStorage};
use lunchbox::{types::ReadableFile, ReadableFileSystem};
use rust_bert::{
    pipelines::{
        common::{ModelResource, ModelType},
        question_answering::{QaInput, QuestionAnsweringConfig, QuestionAnsweringModel},
    },
    resources::LocalResource,
};
use serde::{Deserialize, Serialize};

use crate::{copy_to_local, Model, ModelFromConfig};

/// Config for a question answering model
#[derive(Serialize, Deserialize)]
pub struct CartonQAConfig {
    model_type: ModelType,
    model_path: String,
    config_path: String,
    vocab_path: String,
    merges_path: Option<String>,
    lower_case: bool,
}

pub struct CartonQAModel {
    _tempdir: tempfile::TempDir,
    model: QuestionAnsweringModel,
}

#[async_trait]
impl ModelFromConfig for CartonQAConfig {
    type ModelType = CartonQAModel;

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

        log::trace!("Loading question answering model...");
        // Defaults to cuda if available
        let qa_config = QuestionAnsweringConfig::new(
            self.model_type,
            ModelResource::Torch(td.path().join(self.model_path).into()),
            LocalResource::from(td.path().join(self.config_path)),
            LocalResource::from(td.path().join(self.vocab_path)),
            self.merges_path
                .map(|p| LocalResource::from(td.path().join(p))),
            self.lower_case,
            None,
            None,
        );

        let model = QuestionAnsweringModel::new(qa_config).unwrap();

        CartonQAModel {
            _tempdir: td,
            model,
        }
    }
}

impl Model for CartonQAModel {
    fn infer(&self, tensors: HashMap<String, Tensor>) -> HashMap<String, Tensor> {
        // TODO: don't unwrap
        let question_tensor = tensors.get("question").unwrap();
        let context_tensor = tensors.get("context").unwrap();

        if let Tensor::String(question_tensor) = question_tensor {
            let question_tensor = question_tensor.view();

            // Create an output tensor with the same shape
            let mut output_tensor =
                TensorStorage::new(question_tensor.shape().iter().map(|v| (*v) as _).collect());
            let mut output_view = output_tensor.view_mut();
            let sliced_output_view = output_view.as_slice_mut().unwrap();

            if let Tensor::String(context_tensor) = context_tensor {
                let context_tensor = context_tensor.view();

                // Collect questions and contexts into inputs
                let qa_inputs: Vec<_> = question_tensor
                    .as_slice()
                    .unwrap()
                    .iter()
                    .cloned()
                    .zip(context_tensor.as_slice().unwrap().iter().cloned())
                    .map(|(question, context)| QaInput { question, context })
                    .collect();

                // Run the model and store in output
                // TODO: also provide the score, span start, and end. Also allow setting top_k
                let answers: Vec<_> = self
                    .model
                    .predict(&qa_inputs, 1, 32)
                    .into_iter()
                    .map(|mut answers| answers.pop().unwrap().answer)
                    .collect();
                sliced_output_view.clone_from_slice(&answers);

                let mut out = HashMap::new();
                out.insert("answer".to_owned(), Tensor::String(output_tensor));
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
        info::{DataType, Example, RunnerInfo, Shape, TensorOrMisc, TensorSpec},
        types::{CartonInfo, GenericStorage, PackOpts, Tensor},
    };

    use crate::{download_file, ModelConfig};

    pub async fn pack_distilbert_squad() -> PathBuf {
        let model_config = ModelConfig::QuestionAnswering(super::CartonQAConfig {
            model_type: rust_bert::pipelines::common::ModelType::DistilBert,
            model_path: "./model/rust_model.ot".into(),
            config_path: "./model/config.json".into(),
            vocab_path: "./model/vocab.txt".into(),
            merges_path: None,
            lower_case: false,
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
                "https://huggingface.co/distilbert-base-cased-distilled-squad/resolve/50ba811384f02cb99cdabe5cdc02f7ddc4f69e10/rust_model.ot".into(),
                "8a9f9b2f153ac9ff230aca4548fa3286be9d2f9ea4eb7e9169665b1a8e983f44".into(),
                model_dir.join("rust_model.ot"),
            ),
            download_file(
                "https://huggingface.co/distilbert-base-cased-distilled-squad/resolve/50ba811384f02cb99cdabe5cdc02f7ddc4f69e10/config.json".into(),
                "0b5cb15ec08645604ef7085acfaf9c4131158ac22207a76634574cf2771b1515".into(),
                model_dir.join("config.json"),
            ),
            download_file(
                "https://huggingface.co/distilbert-base-cased-distilled-squad/resolve/50ba811384f02cb99cdabe5cdc02f7ddc4f69e10/vocab.txt".into(),
                "eeaa9875b23b04b4c54ef759d03db9d1ba1554838f8fb26c5d96fa551df93d02".into(),
                model_dir.join("vocab.txt"),
            ),
        );

        // TODO: better error handling
        res.0.unwrap();
        res.1.unwrap();
        res.2.unwrap();

        // Pack the model and return the path
        let info = CartonInfo {
            model_name: Some("DistilBERT base cased distilled SQuAD".into()),
            short_description: Some("A DistilBERT model fine tuned for question answering.".into()),
            model_description: Some("See [here](https://huggingface.co/distilbert-base-cased-distilled-squad) for more details.".into()),
            license: Some("Apache-2.0".into()),
            repository: None,
            homepage: Some("https://huggingface.co/distilbert-base-cased-distilled-squad".into()),
            required_platforms: None,
            inputs: Some(vec![
                TensorSpec {
                    name: "question".into(),
                    dtype: DataType::String,
                    shape: Shape::Symbol("input_shape".into()),
                    description: Some("Questions for the model to answer".into()),
                    internal_name: None
                },
                TensorSpec {
                    name: "context".into(),
                    dtype: DataType::String,
                    shape: Shape::Symbol("input_shape".into()),
                    description: Some("Context for each of the questions. In the same shape as `question`".into()),
                    internal_name: None
                },
            ]),
            outputs: Some(vec![
                TensorSpec {
                    name: "answer".into(),
                    dtype: DataType::String,
                    shape: Shape::Symbol("input_shape".into()),
                    description: Some("Answers to the questions in the same shape as `question`".into()),
                    internal_name: None
                },
            ]),
            self_tests: None,
            examples: Some(vec![
                Example {
                    name: Some("quickstart".into()),
                    description: Some("Answers to some basic questions".into()),
                    inputs: [
                        ("context".into(), TensorOrMisc::Tensor(Tensor::String(ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec!["Amy lives in New Mexico".into(), "While Rob lives in Canada, Bill lives in Egypt".into()]).unwrap().into()).into())),
                        ("question".into(), TensorOrMisc::Tensor(Tensor::String(ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec!["Where does Amy live?".into(), "Where does Bill live?".into()]).unwrap().into()).into())),
                    ].into(),
                    sample_out: [
                        ("answer".into(), TensorOrMisc::Tensor(Tensor::String(ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec!["New Mexico".into(), "Egypt".into()]).unwrap().into()).into()))
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

        carton::Carton::pack::<GenericStorage>(
            dir.path().to_str().unwrap().to_owned(),
            PackOpts {
                info,
                linked_files: None,
            },
        )
        .await
        .unwrap()
    }
}
