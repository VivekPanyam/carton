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

use std::{collections::HashMap, path::Path};

use async_trait::async_trait;
use carton::info::LinkedFile;
use carton_runner_interface::{slowlog::slowlog, types::Tensor};
use lunchbox::{types::ReadableFile, ReadableFileSystem};
use masked_language::CartonMaskedLanguageConfig;
use qa::CartonQAConfig;
use sentiment_analysis::CartonSentimentAnalysisConfig;
use serde::{Deserialize, Serialize};
use summarize::CartonSummarizationConfig;
use text_generation::CartonTextGenerationConfig;
use tokio::io::{AsyncWriteExt, BufReader, BufWriter};
use translate::CartonTranslationConfig;
use zero_shot::CartonZeroShotConfig;

pub mod masked_language;
pub mod qa;
pub mod sentiment_analysis;
pub mod summarize;
pub mod text_generation;
pub mod translate;
pub mod zero_shot;

#[derive(Serialize, Deserialize)]
pub enum ModelConfig {
    Translation(CartonTranslationConfig),
    Summarization(CartonSummarizationConfig),
    ZeroShotClassification(CartonZeroShotConfig),
    SentimentAnalysis(CartonSentimentAnalysisConfig),
    NER,
    POSTagging,
    QuestionAnswering(CartonQAConfig),
    KeywordExtraction,
    TextClassification,
    FillMask(CartonMaskedLanguageConfig),
    SentenceEmbeddings,
    TextGeneration(CartonTextGenerationConfig),
}

#[async_trait]
pub trait ModelFromConfig {
    type ModelType: Model;

    async fn load<F>(self, fs: &F) -> Self::ModelType
    where
        F: ReadableFileSystem + Send + Sync,
        F::FileType: ReadableFile + Unpin + Send + Sync;
}

pub trait Model {
    fn infer(&self, tensors: HashMap<String, Tensor>) -> HashMap<String, Tensor>;
}

pub(crate) async fn copy_to_local<F>(fs: &F, base: &Path, path: &str)
where
    F: ReadableFileSystem,
    F::FileType: ReadableFile + Unpin,
{
    let p = Path::new(path);

    // Create intermediate dirs as necessary
    if let Some(parent_dir) = p.parent() {
        tokio::fs::create_dir_all(base.join(parent_dir))
            .await
            .unwrap();
    }

    let mut sl = slowlog(format!("Loading file '{path}'"), 5)
        .await
        .without_progress();
    let f = fs.open(path).await.unwrap();
    let out = tokio::fs::File::create(base.join(path)).await.unwrap();

    // 1mb buffer
    let mut br = BufReader::with_capacity(1_000_000, f);
    let mut bw = BufWriter::with_capacity(1_000_000, out);

    // TODO: don't unwrap
    tokio::io::copy(&mut br, &mut bw).await.unwrap();
    bw.flush().await.unwrap();
    sl.done();
}

pub(crate) async fn download_file<P: AsRef<std::path::Path>>(
    info: LinkedFile,
    download_path: P,
) -> carton_utils::error::Result<LinkedFile> {
    let url = info.urls.first().unwrap();
    let sha256 = &info.sha256;
    let mut sl = slowlog(format!("Downloading file '{url}'"), 5).await;
    let out = carton_utils::download::cached_download(
        url,
        sha256,
        Some(download_path),
        None,
        |total| {
            if let Some(size) = total {
                sl.set_total(Some(bytesize::ByteSize(size)));
            }
        },
        |downloaded| {
            sl.set_progress(Some(bytesize::ByteSize(downloaded)));
        },
    )
    .await;

    // Let the logging task know we're done downloading
    sl.done();

    out.map(|_| info)
}
