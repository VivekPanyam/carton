use std::{path::PathBuf, time::SystemTime};

use carton::{
    info::RunnerInfo,
    types::{CartonInfo, GenericStorage, LoadOpts, RunnerOpt},
    Carton,
};
use carton_runner_packager::{discovery::RunnerInfo as PackageInfo, DownloadItem};
use semver::VersionReq;
use tokio::process::Command;

// Include the list of python releases we want to build against
include!("../src/bin/build_releases/python_versions.rs");

#[tokio::test]
async fn test_pack_python_model() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    // Build the runner for a specific version of python
    let target_version = PYTHON_VERSIONS.first().unwrap();
    let runner_path = escargot::CargoBuild::new()
        .package("carton-runner-py")
        .bin("carton-runner-py")
        .env(
            "PYO3_CONFIG_FILE",
            manifest_dir.join(format!(
                "python_configs/cpython{}.{}.{}",
                target_version.major, target_version.minor, target_version.micro
            )),
        )
        .current_release()
        .current_target()
        .run()
        .unwrap()
        .path()
        .display()
        .to_string();
    println!("Runner Path: {}", runner_path);

    // Patch the runner on mac
    #[cfg(target_os = "macos")]
    assert!(Command::new("install_name_tool")
        .args(&[
            "-change",
            &format!(
                "/install/lib/libpython{}.{}.dylib",
                target_version.major, target_version.minor
            ),
            &format!(
                "@rpath/libpython{}.{}.dylib",
                target_version.major, target_version.minor
            ),
            &runner_path,
        ])
        .status()
        .await
        .unwrap()
        .success());

    // Create a tempdir to store packaging artifacts
    let tempdir = tempfile::tempdir().unwrap();
    let tempdir_path = tempdir.path();

    let package = carton_runner_packager::package(
        PackageInfo {
            runner_name: "python".to_string(),
            framework_version: semver::Version::new(
                target_version.major as _,
                target_version.minor as _,
                target_version.micro as _,
            ),
            runner_compat_version: 1,
            runner_interface_version: 1,
            runner_release_date: SystemTime::now().into(),
            runner_path,
            platform: target_lexicon::HOST.to_string(),
        },
        vec![DownloadItem {
            url: target_version.url.to_string(),
            sha256: target_version.sha256.to_string(),
            relative_path: "./bundled_python".to_string(),
        }],
    )
    .await;

    // Write the zip file to our temp dir
    let path = tempdir_path.join("runner.zip");
    tokio::fs::write(&path, package.get_data()).await.unwrap();
    let download_info = package.get_download_info(path.to_str().unwrap().to_owned());

    // Now install the runner we just packaged into a tempdir
    let runner_dir = tempfile::tempdir().unwrap();
    std::env::set_var("CARTON_RUNNER_DIR", runner_dir.path());
    carton_runner_packager::install(download_info, true).await;

    let pack_opts: CartonInfo<GenericStorage> = CartonInfo {
        model_name: None,
        short_description: None,
        model_description: None,
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
                ]
                .into(),
            ),
        },
        misc_files: None,
    };

    // Create a "model" with a dependency
    let model_dir = tempfile::tempdir().unwrap();
    tokio::fs::write(model_dir.path().join("requirements.txt"), "xgboost==1.7.3")
        .await
        .unwrap();

    tokio::fs::write(
        model_dir.path().join("main.py"),
        r#"
import xgboost as xgb

class Model:
    def __init__(self):
        pass

    def infer_with_tensors(self, tensors):
        pass

def get_model():
    print("Loaded python model!")
    expected_xgb_version = "1.7.3"
    if xgb.__version__ != expected_xgb_version:
        raise ValueError(f"Got an unexpected version of xgboost. Got {xgb.__version__} and expected {expected_xgb_version}")

    return Model()
"#,
    )
    .await
    .unwrap();

    // Pack and load the model
    let _model = Carton::load_unpacked(
        model_dir.path().to_str().unwrap().to_owned(),
        pack_opts,
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
}
