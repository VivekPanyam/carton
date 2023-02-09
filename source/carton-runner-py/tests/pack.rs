use carton::{
    info::RunnerInfo,
    types::{CartonInfo, GenericStorage, LoadOpts},
    Carton,
};
use semver::VersionReq;

#[tokio::test]
async fn test_pack_python_model() {
    // Make sure the py runner is built
    let runner_path = escargot::CargoBuild::new()
        .package("carton-runner-py")
        .run()
        .unwrap()
        .path()
        .display()
        .to_string();
    println!("Runner Path: {}", runner_path);

    println!("Creating runner.toml");
    let runner_toml = format!(
        r#"
# This is a runner.toml that runs against the release runner
version = 1

[[runner]]
runner_name = "python"
framework_version = "1.0.0"
runner_compat_version = 1
runner_interface_version = 1
runner_release_date = "1979-05-27T07:32:00Z"

# A path to the runner binary. This can be absolute or relative to this file
runner_path = "{runner_path}"

# A target triple
platform = "{}"
"#,
        target_lexicon::HOST.to_string()
    );

    let tempdir = tempfile::tempdir().unwrap();
    std::fs::write(tempdir.path().join("runner.toml"), runner_toml).unwrap();

    // TODO don't do this
    std::env::set_var("CARTON_RUNNER_DIR", tempdir.path().as_os_str());

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
            opts: None,
        },
        misc_files: None,
    };

    // Create a "model" with a dependency
    let model_dir = tempfile::tempdir().unwrap();
    tokio::fs::write(model_dir.path().join("requirements.txt"), "xgboost==1.7.3")
        .await
        .unwrap();

    // TODO: use `pack` instead of `load_unpacked` once `carton::format::v1::save` is implemented
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
