use std::path::PathBuf;

use carton::{
    info::RunnerInfo,
    types::{GenericStorage, LoadOpts, Tensor, TypedStorage},
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
    let builder_path = PathBuf::from(env!("CARGO_BIN_EXE_build_torch_releases"));

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
    let model_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/test_model.pt");

    let packed_model = carton::Carton::pack(
        model_path.to_str().unwrap(),
        RunnerInfo {
            runner_name: "torchscript".into(),
            required_framework_version: semver::VersionReq::parse(">= 2.0.0").unwrap(),
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
    let tensor_a = ndarray::ArrayD::from_shape_vec(vec![1], vec![1.5]).unwrap();
    let tensor_b = ndarray::ArrayD::from_shape_vec(vec![], vec!["scalar".to_owned()]).unwrap();
    let tensor_c =
        ndarray::ArrayD::from_shape_vec(vec![2], vec!["a".to_owned(), "b".to_owned()]).unwrap();
    let out = model
        .infer([
            ("a", Tensor::<GenericStorage>::Float(tensor_a)),
            ("b", Tensor::<GenericStorage>::String(tensor_b)),
            ("c", Tensor::<GenericStorage>::String(tensor_c)),
        ])
        .await
        .unwrap();

    if let Tensor::Float(item) = out.get("doubled").unwrap() {
        assert_eq!(item.view().len(), 1);
        assert_eq!(item.view().ndim(), 1);
        assert_eq!(item.view().first(), Some(&3.0));
    } else {
        panic!("Got an unexpected tensor type for `doubled`")
    }

    if let Tensor::String(item) = out.get("string").unwrap() {
        assert_eq!(item.view().len(), 1);
        assert_eq!(item.view().ndim(), 0);
        assert_eq!(item.view().first(), Some(&"A string".to_owned()));
    } else {
        panic!("Got an unexpected tensor type for `string`")
    }

    if let Tensor::String(item) = out.get("stringlist").unwrap() {
        assert_eq!(item.view().len(), 4);
        assert_eq!(item.view().ndim(), 1);
        assert_eq!(
            item.view().as_slice().unwrap(),
            &["A", "list", "of", "strings"]
                .into_iter()
                .to_owned()
                .collect::<Vec<_>>()
        );
    } else {
        panic!("Got an unexpected tensor type for `stringlist`")
    }
}
