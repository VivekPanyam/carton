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
    let builder_path = PathBuf::from(env!("CARGO_BIN_EXE_build_wasm_releases"));

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
    let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/test_model/model.wasm");

    let packed_model = carton::Carton::pack(
        model_path.to_str().unwrap(),
        RunnerInfo {
            runner_name: "wasm".into(),
            required_framework_version: semver::VersionReq::parse("=0.0.1").unwrap(),
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

    let tensor_in1 = ndarray::ArrayD::from_shape_vec(vec![20], vec![1.5f32; 20]).unwrap();

    let out = model
        .infer([("in1", Tensor::new(tensor_in1))])
        .await
        .unwrap();

    let s = match out.get("out1").unwrap() {
        Tensor::Float(s) => s,
        _ => panic!("Invalid tensor type"),
    };

    assert_eq!(s.view().as_slice().unwrap(), &vec![2.5f32; 20]);
}
