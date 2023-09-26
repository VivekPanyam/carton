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
        summarization::{SummarizationConfig, SummarizationModel},
    },
    resources::LocalResource,
};
use serde::{Deserialize, Serialize};

use crate::{copy_to_local, Model, ModelFromConfig};

/// Config for a summarization model
#[derive(Serialize, Deserialize)]
pub struct CartonSummarizationConfig {
    model_type: ModelType,
    model_path: String,
    config_path: String,
    vocab_path: String,
    merges_path: Option<String>,
}

pub struct CartonSummarizationModel {
    _tempdir: tempfile::TempDir,
    model: SummarizationModel,
}

#[async_trait]
impl ModelFromConfig for CartonSummarizationConfig {
    type ModelType = CartonSummarizationModel;

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

        log::trace!("Loading summarization model...");
        // Defaults to cuda if available
        let summarization_config = SummarizationConfig::new(
            self.model_type,
            ModelResource::Torch(td.path().join(self.model_path).into()),
            LocalResource::from(td.path().join(self.config_path)),
            LocalResource::from(td.path().join(self.vocab_path)),
            self.merges_path
                .map(|p| LocalResource::from(td.path().join(p))),
        );

        let model = SummarizationModel::new(summarization_config).unwrap();

        CartonSummarizationModel {
            _tempdir: td,
            model,
        }
    }
}

impl Model for CartonSummarizationModel {
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

            // Summarize and store in the output
            let summaries = self.model.summarize(input_tensor.as_slice().unwrap());
            sliced_output_view.clone_from_slice(&summaries);

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
        info::{DataType, Example, LinkedFile, RunnerInfo, Shape, TensorOrMisc, TensorSpec},
        types::{CartonInfo, GenericStorage, PackOpts, Tensor},
    };

    use crate::{download_file, ModelConfig};

    // From https://www.nasa.gov/feature/goddard/2023/webb-reveals-colors-of-earendel-most-distant-star-ever-detected
    const SAMPLE_ARTICLE: &'static str = r#"
NASA’s James Webb Space Telescope has followed up on observations by the Hubble Space Telescope of the farthest star ever detected in the very distant universe, within the first billion years after the big bang. Webb’s NIRCam (Near-Infrared Camera) instrument reveals the star to be a massive B-type star more than twice as hot as our Sun, and about a million times more luminous.

The star, which the research team has dubbed Earendel, is located in the Sunrise Arc galaxy and is detectable only due to the combined power of human technology and nature via an effect called gravitational lensing. Both Hubble and Webb were able to detect Earendel due to its lucky alignment behind a wrinkle in space-time created by the massive galaxy cluster WHL0137-08. The galaxy cluster, located between us and Earendel, is so massive that it warps the fabric of space itself, which produces a magnifying effect, allowing astronomers to look through the cluster like a magnifying glass.  

While other features in the galaxy appear multiple times due to the gravitational lensing, Earendel only appears as a single point of light even in Webb’s high-resolution infrared imaging. Based on this, astronomers determine the object is magnified by a factor of at least 4,000, and thus is extremely small – the most distant star ever detected, observed 1 billion years after the big bang. The previous record-holder for the most distant star was detected by Hubble and observed around 4 billion years after the big bang. Another research team using Webb recently identified a gravitationally lensed star they nicknamed Quyllur, a red giant star observed 3 billion years after the big bang.

Stars as massive as Earendel often have companions. Astronomers did not expect Webb to reveal any companions of Earendel since they would be so close together and indistinguishable on the sky. However, based solely on the colors of Earendel, astronomers think they see hints of a cooler, redder companion star. This light has been stretched by the expansion of the universe to wavelengths longer than Hubble’s instruments can detect, and so was only detectable with Webb.

Webb’s NIRCam also shows other notable details in the Sunrise Arc, which is the most highly magnified galaxy yet detected in the universe’s first billion years. Features include both young star-forming regions and older established star clusters as small as 10 light-years across. On either side of the wrinkle of maximum magnification, which runs right through Earendel, these features are mirrored by the distortion of the gravitational lens. The region forming stars appears elongated, and is estimated to be less than 5 million years old. Smaller dots on either side of Earendel are two images of one older, more established star cluster, estimated to be at least 10 million years old. Astronomers determined this star cluster is gravitationally bound and likely to persist until the present day. This shows us how the globular clusters in our own Milky Way might have looked when they formed 13 billion years ago.

Astronomers are currently analyzing data from Webb’s NIRSpec (Near-Infrared Spectrograph) instrument observations of the Sunrise Arc galaxy and Earendel, which will provide precise composition and distance measurements for the galaxy.

Since Hubble’s discovery of Earendel, Webb has detected other very distant stars using this technique, though none quite as far as Earendel. The discoveries have opened a new realm of the universe to stellar physics, and new subject matter to scientists studying the early universe, where once galaxies were the smallest detectable cosmic objects. The research team has cautious hope that this could be a step toward the eventual detection of one of the very first generation of stars, composed only of the raw ingredients of the universe created in the big bang – hydrogen and helium. 
"#;

    // Generated by running the model
    const SAMPLE_SUMMARY: &'static str = "NASA’s James Webb Space Telescope has followed up on observations by the Hubble Space Telescope of the farthest star ever detected. The star, which the research team has dubbed Earendel, is located in the Sunrise Arc galaxy and is detectable only due to the combined power of human technology and nature.";

    pub async fn pack_bart_cnn_dm() -> PathBuf {
        let model_config = ModelConfig::Summarization(super::CartonSummarizationConfig {
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
                    urls: vec!["https://huggingface.co/facebook/bart-large-cnn/resolve/3d224934c6541b2b9147e023c2f6f6fe49bd27e1/rust_model.ot".into()],
                    sha256: "cd0d1586babffa4e90ca71e230290b55b8ebf634319a1c4200c8506ddbae0ab0".into(),
                },
                model_dir.join("rust_model.ot"),
            ),
            download_file(
                LinkedFile {
                    urls: vec!["https://huggingface.co/facebook/bart-large-cnn/resolve/3d224934c6541b2b9147e023c2f6f6fe49bd27e1/config.json".into()],
                    sha256: "c6cb642aec929b65f514ee0ec7c04f9de19f705c143491577ecd8b7cc923c6ed".into(),
                },
                model_dir.join("config.json"),
            ),
            download_file(
                LinkedFile {
                    urls: vec!["https://huggingface.co/facebook/bart-large-cnn/resolve/3d224934c6541b2b9147e023c2f6f6fe49bd27e1/vocab.json".into()],
                    sha256: "9e7f63c2d15d666b52e21d250d2e513b87c9b713cfa6987a82ed89e5e6e50655".into(),
                },
                model_dir.join("vocab.json"),
            ),
            download_file(
                LinkedFile {
                    urls: vec!["https://huggingface.co/facebook/bart-large-cnn/resolve/3d224934c6541b2b9147e023c2f6f6fe49bd27e1/merges.txt".into()],
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
        let info = CartonInfo::<GenericStorage> {
            model_name: Some("BART".into()),
            short_description: Some("A BART model fine-tuned on CNN/Daily Mail to summarize text.".into()),
            model_description: Some("See [here](https://github.com/facebookresearch/fairseq/blob/main/examples/bart/README.md) for more details.".into()),
            license: Some("MIT".into()),
            repository: None,
            homepage: Some("https://github.com/facebookresearch/fairseq/tree/main/examples/bart".into()),
            required_platforms: None,
            inputs: Some(vec![
                TensorSpec {
                    name: "input".into(),
                    dtype: DataType::String,
                    shape: Shape::Symbol("input_shape".into()),
                    description: Some("The strings to summarize".into()),
                    internal_name: None
                },
            ]),
            outputs: Some(vec![
                TensorSpec {
                    name: "output".into(),
                    dtype: DataType::String,
                    shape: Shape::Symbol("input_shape".into()),
                    description: Some("The summarized strings in the same shape as `input`".into()),
                    internal_name: None
                },
            ]),
            self_tests: None,
            examples: Some(vec![
                Example {
                    name: Some("quickstart".into()),
                    description: Some("Summary of a NASA press release".into()),
                    inputs: [
                        ("input".into(), TensorOrMisc::Tensor(Tensor::String(ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1]), vec![SAMPLE_ARTICLE.into()]).unwrap().into()).into())),
                    ].into(),
                    sample_out: [
                        ("output".into(), TensorOrMisc::Tensor(Tensor::String(ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1]), vec![SAMPLE_SUMMARY.into()]).unwrap().into()).into()))
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
