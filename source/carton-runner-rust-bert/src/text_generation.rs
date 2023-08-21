use std::collections::HashMap;

use async_trait::async_trait;
use carton_runner_interface::types::{Tensor, TensorStorage};
use lunchbox::{types::ReadableFile, ReadableFileSystem};
use rust_bert::{
    pipelines::{
        common::{ModelResource, ModelType},
        text_generation::{TextGenerationConfig, TextGenerationModel},
    },
    resources::LocalResource,
};
use serde::{Deserialize, Serialize};

use crate::{copy_to_local, Model, ModelFromConfig};

/// Config for a text generation model
#[derive(Serialize, Deserialize)]
pub struct CartonTextGenerationConfig {
    model_type: ModelType,
    model_path: String,
    config_path: String,
    vocab_path: String,
    merges_path: Option<String>,
}

pub struct CartonTextGenerationModel {
    _tempdir: tempfile::TempDir,
    model: TextGenerationModel,
}

#[async_trait]
impl ModelFromConfig for CartonTextGenerationConfig {
    type ModelType = CartonTextGenerationModel;

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

        log::trace!("Loading text generation model...");
        // Defaults to cuda if available
        let text_generation_config = TextGenerationConfig::new(
            self.model_type,
            ModelResource::Torch(td.path().join(self.model_path).into()),
            LocalResource::from(td.path().join(self.config_path)),
            LocalResource::from(td.path().join(self.vocab_path)),
            self.merges_path
                .map(|p| LocalResource::from(td.path().join(p))),
        );

        let model = TextGenerationModel::new(text_generation_config).unwrap();

        CartonTextGenerationModel {
            _tempdir: td,
            model,
        }
    }
}

impl Model for CartonTextGenerationModel {
    fn infer(&self, tensors: HashMap<String, Tensor>) -> HashMap<String, Tensor> {
        // TODO: don't unwrap
        let input_tensor = tensors.get("input").unwrap();

        if let Tensor::String(input_tensor) = input_tensor {
            let input_tensor = input_tensor.view();

            // Create an output tensor with the same shape
            let mut output_tensor =
                TensorStorage::new(input_tensor.shape().iter().map(|v| (*v) as _).collect());
            let mut output_view = output_tensor.view_mut();
            let sliced_output_view = output_view.as_slice_mut().unwrap();

            // Generate text and store in the output
            let generated_text = self.model.generate(input_tensor.as_slice().unwrap(), None);
            sliced_output_view.clone_from_slice(&generated_text);

            let mut out = HashMap::new();
            out.insert("output".to_owned(), Tensor::String(output_tensor));
            return out;
        }

        // TODO: don't do this
        panic!("Unexpected input");
    }
}

pub mod pack {
    use std::path::PathBuf;

    use carton::{
        info::{DataType, Example, RunnerInfo, Shape, TensorOrMisc, TensorSpec},
        types::{GenericStorage, PackOpts, Tensor},
    };

    use crate::{download_file, ModelConfig};

    pub async fn pack_gpt2_medium() -> PathBuf {
        let model_config = ModelConfig::TextGeneration(super::CartonTextGenerationConfig {
            model_type: rust_bert::pipelines::common::ModelType::GPT2,
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
                "https://huggingface.co/gpt2-medium/resolve/f65d4965d1221eff2bcf34f53a2ba12120e18f24/rust_model.ot".into(),
                "064e9fde8e3a539c41b186a6ca94e6fb7c6520f49f903fb236f6e89912fedd32".into(),
                model_dir.join("rust_model.ot"),
            ),
            download_file(
                "https://huggingface.co/gpt2-medium/resolve/f65d4965d1221eff2bcf34f53a2ba12120e18f24/config.json".into(),
                "ef1a44d889ad1a0acc7731c78134f1b87d2d222f110e97dd10fd4117331caf22".into(),
                model_dir.join("config.json"),
            ),
            download_file(
                "https://huggingface.co/gpt2-medium/resolve/f65d4965d1221eff2bcf34f53a2ba12120e18f24/vocab.json".into(),
                "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783".into(),
                model_dir.join("vocab.json"),
            ),
            download_file(
                "https://huggingface.co/gpt2-medium/resolve/f65d4965d1221eff2bcf34f53a2ba12120e18f24/merges.txt".into(),
                "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5".into(),
                model_dir.join("merges.txt"),
            ),
        );

        // TODO: better error handling
        res.0.unwrap();
        res.1.unwrap();
        res.2.unwrap();
        res.3.unwrap();

        // Pack the model and return the path
        carton::Carton::pack::<GenericStorage>(dir.path().to_str().unwrap().to_owned(), PackOpts {
            model_name: Some("GPT2 Medium".into()),
            short_description: Some("GPT2 Medium".into()),
            model_description: Some("See [here](https://github.com/openai/gpt-2) for more details.".into()),
            required_platforms: None,
            inputs: Some(vec![
                TensorSpec {
                    name: "input".into(),
                    dtype: DataType::String,
                    shape: Shape::Symbol("input_shape".into()),
                    description: Some("The prompts to pass to the model".into()),
                    internal_name: None
                },
            ]),
            outputs: Some(vec![
                TensorSpec {
                    name: "output".into(),
                    dtype: DataType::String,
                    shape: Shape::Symbol("input_shape".into()),
                    description: Some("The continued strings in the same shape as `input`".into()),
                    internal_name: None
                },
            ]),
            self_tests: None,
            examples: Some(vec![
                Example {
                    name: Some("quickstart".into()),
                    description: Some("Example continuations".into()),
                    inputs: [
                        ("input".into(), TensorOrMisc::Tensor(Tensor::String(ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec!["The dog".into(), "The cat was".into()]).unwrap().into()).into())),
                    ].into(),
                    sample_out: [
                        ("output".into(), TensorOrMisc::Tensor(Tensor::String(ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec!["The dog's owner, who did not want to be identified, said the dog was well behaved ...".into(), "The cat was not allowed to leave the house. She was placed in the care of ...".into()]).unwrap().into()).into()))
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
        }).await.unwrap()
    }
}
