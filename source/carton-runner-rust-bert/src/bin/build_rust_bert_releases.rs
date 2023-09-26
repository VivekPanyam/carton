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
        .package("carton-runner-rust-bert")
        .bin("carton-runner-rust-bert")
        .current_release()
        .current_target()
        .arg("--timings")
        // TODO: remove when https://github.com/rust-lang/cargo/issues/12434 is fixed
        .env_remove("LIBTORCH")
        .env_remove("LIBTORCH_BYPASS_VERSION_CHECK")
        .run()
        .unwrap()
        .path()
        .display()
        .to_string();
    sl.done();
    log::info!("Runner Path: {}", runner_path);

    let package = carton_runner_packager::package(
        RunnerInfo {
            runner_name: "rust-bert".to_string(),
            framework_version: semver::Version::new(0, 21, 0),
            runner_compat_version: 1,
            runner_interface_version: 1,
            runner_release_date: SystemTime::now().into(),
            runner_path,
            platform: target_lexicon::HOST.to_string(),
        },
        vec![DownloadItem {
            url: fetch_deps::libtorch::URL.to_string(),
            sha256: fetch_deps::libtorch::SHA256.to_string(),

            // The zip file includes a libtorch folder
            relative_path: "./".to_string(),
        }],
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
