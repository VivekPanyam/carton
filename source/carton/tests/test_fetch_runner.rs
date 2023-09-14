//! This test tests fetching and installation of a runner

use carton::{
    info::RunnerInfo,
    types::{CartonInfo, GenericStorage, LoadOpts, PackOpts, RunnerOpt},
    Carton,
};
use semver::VersionReq;

#[tokio::test]
async fn main() {
    // Create a new directory to store runners in
    let runner_dir = tempfile::tempdir().unwrap();
    std::env::set_var("CARTON_RUNNER_DIR", runner_dir.path());

    // Pack a model that requires a specific version of python
    let info: CartonInfo<GenericStorage> = CartonInfo {
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
            required_framework_version: VersionReq::parse("=3.11").unwrap(),
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

    // Create a "model"
    let model_dir = tempfile::tempdir().unwrap();
    tokio::fs::write(model_dir.path().join("requirements.txt"), "")
        .await
        .unwrap();

    tokio::fs::write(
        model_dir.path().join("main.py"),
        r#"
import sys

class Model:
    def __init__(self):
        pass

    def infer_with_tensors(self, tensors):
        pass

def get_model():
    print("Loaded python model!")
    if sys.version_info.major != 3 or sys.version_info.minor != 11:
        raise ValueError(f"Got an unexpected version of python. Got {sys.version_info.major}.{sys.version_info.minor} and expected 3.11")

    return Model()
"#,
    )
    .await
    .unwrap();

    // Pack and load the model
    let _model = Carton::load_unpacked(
        model_dir.path().to_str().unwrap().to_owned(),
        PackOpts {
            info,
            linked_files: None,
        },
        LoadOpts::default(),
    )
    .await
    .unwrap();

    let dir = std::fs::read_dir(runner_dir.path()).unwrap();
    assert!(dir.into_iter().count() > 0);
}
