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

use std::path::PathBuf;

use carton::{
    info::RunnerInfo,
    types::{LoadOpts, Tensor},
};
use carton_runner_packager::RunnerPackage;
use tokio::process::Command;

#[tokio::test]
async fn test_pack() {
    // Logging (for long running downloads)
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .is_test(true)
        .init();

    // Get the path of the builder
    let builder_path = PathBuf::from(env!("CARGO_BIN_EXE_build_xla_releases"));

    // Create a tempdir to store packaging artifacts
    let tempdir = tempfile::tempdir().unwrap();
    let tempdir_path = tempdir.path();

    // Run the builder
    let status = Command::new(builder_path)
        .args(&["--output-path", tempdir_path.to_str().unwrap()])
        .status()
        .await
        .unwrap();
    assert!(status.success());

    // Get a package
    let package_config = std::fs::read_dir(&tempdir_path)
        .unwrap()
        .find_map(|item| {
            if let Ok(item) = item {
                if item.file_name().to_str().unwrap().ends_with(".json") {
                    return Some(item);
                }
            }

            None
        })
        .unwrap();

    let package: RunnerPackage =
        serde_json::from_slice(&std::fs::read(package_config.path()).unwrap()).unwrap();

    // Get the zipfile path
    let path = tempdir_path.join(format!("{}.zip", package.get_data_sha256()));
    let download_info = package.get_download_info(path.to_str().unwrap().to_owned());

    // Now install the runner we just packaged into a tempdir
    let runner_dir = tempfile::tempdir().unwrap();
    std::env::set_var("CARTON_RUNNER_DIR", runner_dir.path());
    log::info!("About to install runner");
    carton_runner_packager::install(download_info, true).await;
    log::info!("Installed runner");

    // Pack a model
    let model_dir = tempfile::tempdir().unwrap();
    let model_file_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/test_model.pb");
    let model_info_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/test_model.json");

    tokio::fs::copy(model_file_path, model_dir.path().join("model.pb"))
        .await
        .unwrap();
    tokio::fs::copy(model_info_path, model_dir.path().join("model.json"))
        .await
        .unwrap();

    let packed_model = carton::Carton::pack(
        model_dir.path().to_str().unwrap(),
        RunnerInfo {
            runner_name: "xla".into(),
            required_framework_version: semver::VersionReq::parse(">= 0.0.0").unwrap(),
            runner_compat_version: None,
            opts: None,
        },
    )
    .await
    .unwrap();

    // Load the model
    let model = carton::Carton::load(packed_model.to_str().unwrap(), LoadOpts::default())
        .await
        .unwrap();

    // Create an input tensor and run inference
    let tensor_a =
        ndarray::ArrayD::from_shape_vec(vec![5], vec![1.5f32, 3.0, 5.2, 1.0, 2.2]).unwrap();
    let out = model.infer([("a", Tensor::new(tensor_a))]).await.unwrap();

    if let Tensor::Float(item) = out.get("doubled").unwrap() {
        assert_eq!(item.view().len(), 5);
        assert_eq!(item.view().ndim(), 1);
        assert_eq!(item.view().as_slice().unwrap(), [3.0, 6.0, 10.4, 2.0, 4.4]);
    } else {
        panic!("Got an unexpected tensor type for `doubled`")
    }

    if let Tensor::Float(item) = out.get("tripled").unwrap() {
        assert_eq!(item.view().len(), 5);
        assert_eq!(item.view().ndim(), 1);
        assert_eq!(item.view().as_slice().unwrap(), [4.5, 9.0, 15.6, 3.0, 6.6]);
    } else {
        panic!("Got an unexpected tensor type for `tripled`")
    }
}
