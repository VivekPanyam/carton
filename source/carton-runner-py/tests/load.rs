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
    types::{CartonInfo, LoadOpts, PackOpts, RunnerOpt, Tensor},
    Carton,
};
use carton_runner_packager::RunnerPackage;
use futures_util::StreamExt;
use semver::VersionReq;
use tokio::process::Command;

#[tokio::test]
async fn test_pack_python_model() {
    // Get the path of the builder
    let builder_path = PathBuf::from(env!("CARGO_BIN_EXE_build_releases"));

    // Create a tempdir to store packaging artifacts
    let tempdir = tempfile::tempdir().unwrap();
    let tempdir_path = tempdir.path();

    // Run the builder
    let status = Command::new(builder_path)
        .args(&[
            "--output-path",
            tempdir_path.to_str().unwrap(),
            "--single-release",
        ])
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
    carton_runner_packager::install(download_info, true).await;

    let info = CartonInfo {
        model_name: None,
        short_description: None,
        model_description: None,
        license: None,
        repository: None,
        homepage: None,
        required_platforms: None,
        inputs: None,
        outputs: None,
        self_tests: None,
        examples: None,
        runner: RunnerInfo {
            runner_name: "python".into(),
            required_framework_version: VersionReq::parse("*").unwrap(),
            runner_compat_version: None,
            opts: Some(
                [
                    (
                        "entrypoint_package".into(),
                        RunnerOpt::String("main".into()),
                    ),
                    (
                        "entrypoint_fn".into(),
                        RunnerOpt::String("get_model".into()),
                    ),
                    (
                        "model.an_example_custom_option".into(),
                        RunnerOpt::String("some_string_value".into()),
                    ),
                    (
                        "model.another_example_custom_option".into(),
                        RunnerOpt::Boolean(false),
                    ),
                ]
                .into(),
            ),
        },
        misc_files: None,
    };

    // Create a "model" with a dependency
    // This also tests symlinks so it's doing much more than necessary
    // You can just create a normal requirements.txt in the root of the model directory
    let model_dir = tempfile::tempdir().unwrap();
    tokio::fs::write(
        model_dir.path().join("requirements_symlink_target.txt"),
        "xgboost==1.7.3",
    )
    .await
    .unwrap();

    // Test symlinks
    tokio::fs::create_dir(model_dir.path().join("something"))
        .await
        .unwrap();

    // requirements.txt -> something/symlink_one.txt
    tokio::fs::symlink(
        model_dir.path().join("something").join("symlink_one.txt"),
        model_dir.path().join("requirements.txt"),
    )
    .await
    .unwrap();

    // something/symlink_one.txt -> ../requirements_symlink_target.txt
    tokio::fs::symlink(
        "../requirements_symlink_target.txt",
        model_dir.path().join("something").join("symlink_one.txt"),
    )
    .await
    .unwrap();

    tokio::fs::write(
        model_dir.path().join("main.py"),
        r#"
import os
import os.path
import xgboost as xgb
import numpy as np

class Model:
    def __init__(self):
        pass

    async def infer_with_tensors(self, tensors):
        for i in range(5):
            yield {
                "a": np.zeros(i, dtype=np.float32)
            }

def get_model(an_example_custom_option, another_example_custom_option):
    print("Loaded python model!")
    assert os.path.islink("requirements.txt")
    assert os.readlink("requirements.txt").startswith("/")
    expected_xgb_version = "1.7.3"
    if xgb.__version__ != expected_xgb_version:
        raise ValueError(f"Got an unexpected version of xgboost. Got {xgb.__version__} and expected {expected_xgb_version}")

    if an_example_custom_option != "some_string_value":
        raise ValueError("an_example_custom_option did not match the expected value")

    if another_example_custom_option != False:
        raise ValueError("another_example_custom_option did not match the expected value")

    return Model()
"#,
    )
    .await
    .unwrap();

    // Test `load_unpacked`
    let _model = Carton::load_unpacked(
        model_dir.path().to_str().unwrap().to_owned(),
        PackOpts {
            info: info.clone(),
            linked_files: None,
        },
        LoadOpts::default(),
    )
    .await
    .unwrap();

    // Load the generated lockfile
    let lockfile = String::from_utf8(
        tokio::fs::read(&model_dir.path().join(".carton/carton.lock"))
            .await
            .unwrap(),
    )
    .unwrap();

    assert!(lockfile.contains("xgboost"));
    assert!(lockfile.contains("scipy"));
    assert!(lockfile.contains("numpy"));

    // Test pack followed by load
    let packed_path = Carton::pack(
        model_dir.path().to_str().unwrap().to_owned(),
        PackOpts {
            info,
            linked_files: None,
        },
    )
    .await
    .unwrap();

    let model = Carton::load(
        packed_path.to_str().unwrap().to_owned(),
        LoadOpts::default(),
    )
    .await
    .unwrap();

    // Make sure streaming inference works
    let res = model.streaming_infer::<_, &str>([]).await;

    futures_util::pin_mut!(res);

    assert_eq!(
        res.next().await.unwrap().unwrap().get("a").unwrap(),
        &Tensor::new::<f32, _>(ndarray::ArrayD::from_shape_vec(vec![0], vec![]).unwrap())
    );
    assert_eq!(
        res.next().await.unwrap().unwrap().get("a").unwrap(),
        &Tensor::new(ndarray::ArrayD::from_shape_vec(vec![1], vec![0f32]).unwrap())
    );
    assert_eq!(
        res.next().await.unwrap().unwrap().get("a").unwrap(),
        &Tensor::new(ndarray::ArrayD::from_shape_vec(vec![2], vec![0f32, 0.0]).unwrap())
    );
    assert_eq!(
        res.next().await.unwrap().unwrap().get("a").unwrap(),
        &Tensor::new(ndarray::ArrayD::from_shape_vec(vec![3], vec![0f32, 0.0, 0.0]).unwrap())
    );
    assert_eq!(
        res.next().await.unwrap().unwrap().get("a").unwrap(),
        &Tensor::new(ndarray::ArrayD::from_shape_vec(vec![4], vec![0f32, 0.0, 0.0, 0.0]).unwrap())
    );

    assert!(res.next().await.is_none());

    // Make sure regular inference works
    let res = model.infer::<_, &str>([]).await.unwrap();
    assert_eq!(
        res.get("a").unwrap(),
        &Tensor::new(ndarray::ArrayD::from_shape_vec(vec![4], vec![0f32, 0.0, 0.0, 0.0]).unwrap())
    );
}
