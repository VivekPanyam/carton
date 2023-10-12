//! Same as torch runner

use std::{path::PathBuf, time::SystemTime};

use clap::Parser;

use carton_runner_interface::slowlog::slowlog;
use carton_runner_packager::discovery::RunnerInfo;

// TODO: This should be the version of carton-interface-wasm, but it's not done yet.
const INTERFACE_VERSION: semver::Version = semver::Version::new(0, 0, 1);

#[derive(Parser, Debug)]
struct Args {
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
        .package("carton-runner-wasm")
        .bin("carton-runner-wasm")
        .current_release()
        .current_target()
        .arg("--timings")
        .run()
        .unwrap()
        .path()
        .display()
        .to_string();

    sl.done();
    log::info!("Runner Path: {}", runner_path);

    let package = carton_runner_packager::package(
        RunnerInfo {
            runner_name: "wasm".to_string(),
            framework_version: INTERFACE_VERSION,
            runner_compat_version: 1,
            runner_interface_version: 1,
            runner_release_date: SystemTime::now().into(),
            runner_path,
            platform: target_lexicon::HOST.to_string(),
        },
        vec![],
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
