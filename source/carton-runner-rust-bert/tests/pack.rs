use std::{collections::HashMap, path::PathBuf};

use carton::{info::TensorOrMisc, types::LoadOpts};
use carton_runner_packager::RunnerPackage;
use tokio::process::Command;

#[tokio::test]
async fn test_pack() {
    // Logging (for long running downloads)
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .is_test(true)
        .init();

    // Get the path of the builder
    let builder_path = PathBuf::from(env!("CARGO_BIN_EXE_build_rust_bert_releases"));

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

    // Pack models
    let m2m100_path = carton_runner_rust_bert::translate::pack::pack_m2m100().await;
    log::info!("Packed m2m100 model: {m2m100_path:#?}");

    let model = carton::Carton::load(
        m2m100_path.to_str().unwrap().to_owned(),
        LoadOpts {
            visible_device: carton::types::Device::maybe_from_index(0),
            ..Default::default()
        },
    )
    .await
    .unwrap();

    log::info!("Loaded model. Getting example inputs");

    let ex = &model.get_info().info.examples.as_ref().unwrap()[0];
    let mut tensors = HashMap::new();
    for (k, v) in &ex.inputs {
        if let TensorOrMisc::Tensor(t) = v {
            let t = t.get().await.clone();
            tensors.insert(k.clone(), t);
        } else {
            panic!("Expected tensor but got misc");
        }
    }

    log::info!("running inference");
    let out = model.infer_with_inputs(tensors).await.unwrap();
    for (k, v) in out {
        log::info!("{k}: {v:#?}");
    }

    // Delete the packed model
    tokio::fs::remove_file(m2m100_path).await.unwrap();
}
