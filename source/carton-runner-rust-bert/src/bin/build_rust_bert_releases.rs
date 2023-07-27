//! A binary that builds the release packages

use std::{path::PathBuf, time::SystemTime};

use carton_runner_interface::slowlog::slowlog;
use carton_runner_packager::{discovery::RunnerInfo, DownloadItem};
use clap::Parser;

#[derive(Parser, Debug)]
struct Args {
    /// The local folder to output to
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
        .package("carton-runner-rust-bert")
        .bin("carton-runner-rust-bert")
        .current_release()
        .current_target()
        .env_remove("LIBTORCH")
        .run()
        .unwrap()
        .path()
        .display()
        .to_string();
    sl.done();
    log::info!("Runner Path: {}", runner_path);

    let package = carton_runner_packager::package(
        RunnerInfo {
            runner_name: "rust-bert".to_string(),
            framework_version: semver::Version::new(0, 21, 0),
            runner_compat_version: 1,
            runner_interface_version: 1,
            runner_release_date: SystemTime::now().into(),
            runner_path,
            platform: target_lexicon::HOST.to_string(),
        },
        vec![DownloadItem {
            url: fetch_deps::libtorch::URL.to_string(),
            sha256: fetch_deps::libtorch::SHA256.to_string(),

            // The zip file includes a libtorch folder
            relative_path: "./".to_string(),
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
}
