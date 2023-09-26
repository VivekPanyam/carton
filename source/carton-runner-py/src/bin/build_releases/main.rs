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

use carton_runner_packager::{discovery::RunnerInfo, DownloadItem};
use clap::Parser;
use python_versions::{PythonVersion, PYTHON_VERSIONS};
mod python_versions;

#[derive(Parser, Debug)]
struct Args {
    /// The local folder to output to
    #[arg(long)]
    output_path: PathBuf,

    // Only build one release (useful in tests)
    #[arg(long)]
    single_release: bool,
}

#[tokio::main]
async fn main() {
    // Parse args
    let args = Args::parse();

    for PythonVersion {
        url,
        sha256,
        major,
        minor,
        micro,
    } in PYTHON_VERSIONS
    {
        println!("Building runner for python {major}.{minor}.{micro}");
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

        // Build the runner for a specific version of python
        let runner_path = escargot::CargoBuild::new()
            .package("carton-runner-py")
            .bin("carton-runner-py")
            .env(
                "PYO3_CONFIG_FILE",
                manifest_dir.join(format!("python_configs/cpython{major}.{minor}.{micro}")),
            )
            .current_release()
            .current_target()
            .arg("--timings")
            .run()
            .unwrap()
            .path()
            .display()
            .to_string();
        println!("Runner Path: {}", runner_path);

        // Patch the runner on mac
        #[cfg(target_os = "macos")]
        assert!(tokio::process::Command::new("install_name_tool")
            .args(&[
                "-change",
                &format!("/install/lib/libpython{major}.{minor}.dylib",),
                &format!("@rpath/libpython{major}.{minor}.dylib",),
                &runner_path,
            ])
            .status()
            .await
            .unwrap()
            .success());

        let package = carton_runner_packager::package(
            RunnerInfo {
                runner_name: "python".to_string(),
                framework_version: semver::Version::new(major as _, minor as _, micro as _),
                runner_compat_version: 1,
                runner_interface_version: 1,
                runner_release_date: SystemTime::now().into(),
                runner_path,
                platform: target_lexicon::HOST.to_string(),
            },
            vec![DownloadItem {
                url: url.to_string(),
                sha256: sha256.to_string(),
                relative_path: "./bundled_python".to_string(),
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

        if args.single_release {
            break;
        }
    }
}
