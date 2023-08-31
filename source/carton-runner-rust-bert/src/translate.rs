use std::collections::HashMap;

use async_trait::async_trait;
use carton_runner_interface::types::{Tensor, TensorStorage};
use lunchbox::{types::ReadableFile, ReadableFileSystem};
use rust_bert::{
    pipelines::{
        common::{ModelResource, ModelType},
        translation::{Language, TranslationConfig, TranslationModel},
    },
    resources::LocalResource,
};
use serde::{Deserialize, Serialize};

use crate::{copy_to_local, Model, ModelFromConfig};

/// Config for a translation model
#[derive(Serialize, Deserialize)]
pub struct CartonTranslationConfig {
    model_type: ModelType,
    model_path: String,
    config_path: String,
    vocab_path: String,
    merges_path: Option<String>,
    source_languages: Vec<Language>,
    target_languages: Vec<Language>,
}

pub struct CartonTranslationModel {
    _tempdir: tempfile::TempDir,
    model: TranslationModel,
}

#[async_trait]
impl ModelFromConfig for CartonTranslationConfig {
    type ModelType = CartonTranslationModel;

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

        log::trace!("Loading translation model...");
        let translation_config = TranslationConfig::new(
            self.model_type,
            ModelResource::Torch(td.path().join(self.model_path).into()),
            LocalResource::from(td.path().join(self.config_path)),
            LocalResource::from(td.path().join(self.vocab_path)),
            self.merges_path
                .map(|p| LocalResource::from(td.path().join(p))),
            self.source_languages,
            self.target_languages,
            // Defaults to cuda if available
            None,
        );

        let model = TranslationModel::new(translation_config).unwrap();

        CartonTranslationModel {
            _tempdir: td,
            model,
        }
    }
}

impl Model for CartonTranslationModel {
    fn infer(&self, tensors: HashMap<String, Tensor>) -> HashMap<String, Tensor> {
        // TODO: don't unwrap
        let input_tensor = tensors.get("input").unwrap();
        let source_language = tensors.get("source_language").unwrap();
        let target_language = tensors.get("target_language").unwrap();

        // Get all of them as string tensors
        if let Tensor::String(input_tensor) = input_tensor {
            let input_tensor = input_tensor.view();
            let mut output_tensor =
                TensorStorage::new(input_tensor.shape().iter().map(|v| (*v) as _).collect());
            let mut output_view = output_tensor.view_mut();

            if let Tensor::String(source_language) = source_language {
                let source_language = source_language.view();

                if let Tensor::String(target_language) = target_language {
                    let target_language = target_language.view();

                    for batch_idx in 0..input_tensor.len_of(ndarray::Axis(0)) {
                        let indexed_input_tensor =
                            input_tensor.index_axis(ndarray::Axis(0), batch_idx);
                        let data = indexed_input_tensor.as_slice().unwrap();
                        let sl = source_language.get(batch_idx).unwrap();
                        let tl = target_language.get(batch_idx).unwrap();

                        let result = self
                            .model
                            .translate(
                                data,
                                serde_plain::from_str::<Option<Language>>(sl).unwrap(),
                                serde_plain::from_str::<Option<Language>>(tl).unwrap(),
                            )
                            .unwrap();
                        log::trace!(
                            "Translation: {data:#?} from {sl} to {tl} provides {result:#?}"
                        );

                        // Set the values of the output tensor
                        let mut indexed_output_view =
                            output_view.index_axis_mut(ndarray::Axis(0), batch_idx);
                        let sliced_output_view = indexed_output_view.as_slice_mut().unwrap();
                        sliced_output_view.clone_from_slice(&result);
                    }

                    let mut out = HashMap::new();
                    out.insert("output".to_owned(), Tensor::String(output_tensor));
                    return out;
                }
            }
        }

        // TODO: don't do this
        panic!("Unexpected input");
    }
}

pub mod pack {
    use std::path::PathBuf;

    use carton::{
        info::{DataType, Dimension, Example, RunnerInfo, Shape, TensorOrMisc, TensorSpec},
        types::{CartonInfo, GenericStorage, PackOpts, Tensor},
    };
    use rust_bert::{m2m_100::M2M100SourceLanguages, pipelines::translation::Language};

    use crate::{download_file, ModelConfig};

    pub async fn pack_m2m100() -> PathBuf {
        // Replace ChineseMandarin with Chinese
        let languages: Vec<_> = M2M100SourceLanguages::M2M100_418M
            .iter()
            .map(|item| match item {
                Language::ChineseMandarin => Language::Chinese,
                other => *other,
            })
            .collect();

        let model_config = ModelConfig::Translation(super::CartonTranslationConfig {
            model_type: rust_bert::pipelines::common::ModelType::M2M100,
            model_path: "./model/rust_model.ot".into(),
            config_path: "./model/config.json".into(),
            vocab_path: "./model/vocab.json".into(),
            merges_path: Some("./model/sentencepiece.bpe.model".into()),
            source_languages: languages.clone(),
            target_languages: languages.clone(),
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
                "https://huggingface.co/facebook/m2m100_418M/resolve/a84767a43c9159c5c15eb3964dce2179684647f6/rust_model.ot".into(),
                "f170f6a277d00b20144fa6dac6ecd781c5a5e66844c022244437dd2da3a83655".into(),
                model_dir.join("rust_model.ot"),
            ),
            download_file(
                "https://huggingface.co/facebook/m2m100_418M/resolve/a84767a43c9159c5c15eb3964dce2179684647f6/config.json".into(),
                "df0ae43e4e4b0d7e3c97b7f447857a70ef6b6a2aa1f145cedbcc730d95f67134".into(),
                model_dir.join("config.json"),
            ),
            download_file(
                "https://huggingface.co/facebook/m2m100_418M/resolve/a84767a43c9159c5c15eb3964dce2179684647f6/vocab.json".into(),
                "b6e77e474aeea8f441363aca7614317c06381f3eacfe10fb9856d5081d1074cc".into(),
                model_dir.join("vocab.json"),
            ),
            download_file(
                "https://huggingface.co/facebook/m2m100_418M/resolve/a84767a43c9159c5c15eb3964dce2179684647f6/sentencepiece.bpe.model".into(),
                "d8f7c76ed2a5e0822be39f0a4f95a55eb19c78f4593ce609e2edbc2aea4d380a".into(),
                model_dir.join("sentencepiece.bpe.model"),
            ),
        );

        // TODO: better error handling
        res.0.unwrap();
        res.1.unwrap();
        res.2.unwrap();
        res.3.unwrap();

        // Pack the model and return the path
        let info = CartonInfo {
            model_name: Some("M2M100".into()),
            short_description: Some("M2M100 is a model that can translate directly between any pair of 100 languages.".into()),
            model_description: Some("See [here](https://about.fb.com/news/2020/10/first-multilingual-machine-translation-model/) for more details. M2M100 supports the following languages:\n".to_owned() + &languages.iter().map(|l| format!("- {}", serde_plain::to_string(l).unwrap())).collect::<Vec<_>>().join("\n")),
            required_platforms: None,
            inputs: Some(vec![
                TensorSpec {
                    name: "input".into(),
                    dtype: DataType::String,
                    shape: Shape::Shape(vec![Dimension::Symbol("N".into()), Dimension::Any]),
                    description: Some("The strings to translate as batches grouped by language".into()),
                    internal_name: None
                },
                TensorSpec {
                    name: "source_language".into(),
                    dtype: DataType::String,
                    shape: Shape::Shape(vec![Dimension::Symbol("N".into())]),
                    description: Some("The source language (or empty string) for every batch item".into()),
                    internal_name: None
                },
                TensorSpec {
                    name: "target_language".into(),
                    dtype: DataType::String,
                    shape: Shape::Shape(vec![Dimension::Symbol("N".into())]),
                    description: Some("The target language for every batch item".into()),
                    internal_name: None
                }
            ]),
            outputs: Some(vec![
                TensorSpec {
                    name: "output".into(),
                    dtype: DataType::String,
                    shape: Shape::Shape(vec![Dimension::Symbol("N".into()), Dimension::Any]),
                    description: Some("The translated strings in the same shape as `input`".into()),
                    internal_name: None
                },
            ]),
            self_tests: None,
            examples: Some(vec![
                Example {
                    name: Some("quickstart".into()),
                    description: Some("Translation from Hindi to French and Chinese to English".into()),
                    inputs: [
                        ("input".into(), TensorOrMisc::Tensor(Tensor::String(ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 1]), vec!["जीवन एक चॉकलेट बॉक्स की तरह है।".into(), "生活就像一盒巧克力。".into()]).unwrap().into()).into())),
                        ("source_language".into(), TensorOrMisc::Tensor(Tensor::String(ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec!["Hindi".into(), "Chinese".into()]).unwrap().into()).into())),
                        ("target_language".into(), TensorOrMisc::Tensor(Tensor::String(ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec!["French".into(), "English".into()]).unwrap().into()).into())),
                    ].into(),
                    sample_out: [
                        ("output".into(), TensorOrMisc::Tensor(Tensor::String(ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 1]), vec!["La vie est comme une boîte de chocolat.".into(), "Life is like a box of chocolate.".into()]).unwrap().into()).into()))
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
