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

//! A binary that builds the release packages

use std::{path::PathBuf, time::SystemTime};

use carton_runner_interface::slowlog::slowlog;
use carton_runner_packager::{discovery::RunnerInfo, DownloadItem};
use clap::Parser;

#[derive(Parser, Debug)]
struct Args {
    /// The local folder to output to
    #[arg(long)]
    output_path: PathBuf,
}

#[tokio::main]
async fn main() {
    // Logging (for long running downloads)
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Parse args
    let args = Args::parse();

    log::info!("Starting runner build...");
    let mut sl = slowlog("Building runner", 5).await.without_progress();
    // Build the runner
    let runner_path = escargot::CargoBuild::new()
        .package("carton-runner-xla")
        .bin("carton-runner-xla")
        .current_release()
        .current_target()
        .arg("--timings")
        // TODO: remove when https://github.com/rust-lang/cargo/issues/12434 is fixed
        .env_remove("XLA_EXTENSION_DIR")
        .run()
        .unwrap()
        .path()
        .display()
        .to_string();
    sl.done();
    log::info!("Runner Path: {}", runner_path);

    let package = carton_runner_packager::package(
        RunnerInfo {
            runner_name: "xla".to_string(),
            framework_version: fetch_deps::XLA_VERSION,
            runner_compat_version: 1,
            runner_interface_version: 1,
            runner_release_date: SystemTime::now().into(),
            runner_path,
            platform: target_lexicon::HOST.to_string(),
        },
        vec![
            DownloadItem {
                url: fetch_deps::xla::URL.to_string(),
                sha256: fetch_deps::xla::SHA256.to_string(),

                // The file includes a xla_extension folder
                relative_path: "./".to_string(),
            },
            // Download cuda runtime libs on x86 linux
            #[cfg(all(not(target_os = "macos"), target_arch = "x86_64"))]
            DownloadItem {
                url: "https://files.pythonhosted.org/packages/95/46/6361d45c7a6fe3b3bb8d5fa35eb43c1dcd12d14799a0dc6faef3d76eaf41/nvidia_cuda_runtime_cu12-12.2.140-py3-none-manylinux1_x86_64.whl".into(),
                sha256: "acb9cb7e44594a3512533497ea790d9f0b0e399abcc888584c157d1ffa080e41".into(),
                relative_path: "cudart".into(),
            },
            #[cfg(all(not(target_os = "macos"), target_arch = "x86_64"))]
            DownloadItem {
                url: "https://files.pythonhosted.org/packages/fa/d7/f46bd08337201ec875bda25e29fcb65d26bdc6d0be306dc33fc0b092faa6/nvidia_cudnn_cu12-8.9.4.25-py3-none-manylinux1_x86_64.whl".into(),
                sha256: "9f882d5b11753c566fd32427839089eb4f8ba7a58c28446ad359e32c40229723".into(),
                relative_path: "cudnn".into(),
            },
            #[cfg(all(not(target_os = "macos"), target_arch = "x86_64"))]
            DownloadItem {
                url: "https://files.pythonhosted.org/packages/b6/6a/e8cca34f85b18a0280e3a19faca1923f6a04e7d587e9d8e33bc295a52b6d/nvidia_cublas_cu12-12.2.5.6-py3-none-manylinux1_x86_64.whl".into(),
                sha256: "f448d8db463f9a2c7dd9a1e607c0961d3e95f825b86839c5882f7d073d406df0".into(),
                relative_path: "cublas".into(),
            },
            #[cfg(all(not(target_os = "macos"), target_arch = "x86_64"))]
            DownloadItem {
                url: "https://files.pythonhosted.org/packages/a7/a5/1b48eeda9bdc3ac5bf00d84eca6f31b568ab3da9008f754bf1fdd98ee97b/nvidia_cuda_nvcc_cu12-12.2.140-py3-none-manylinux1_x86_64.whl".into(),
                sha256: "c6ff4ae7a636eb6cab7621191e19ac3779690686bbcf52c04d8b1ca2944e1e27".into(),
                relative_path: "nvcc".into(),
            },
        ],
    )
    .await;

    // Write the zip file to our output dir
    tokio::fs::write(
        &args
            .output_path
            .join(format!("{}.zip", package.get_data_sha256())),
        package.get_data(),
    )
    .await
    .unwrap();

    // Write the package config so it can be loaded when the runner zip files will be uploaded
    tokio::fs::write(
        &args.output_path.join(format!("{}.json", package.get_id())),
        serde_json::to_string_pretty(&package).unwrap(),
    )
    .await
    .unwrap();
}
