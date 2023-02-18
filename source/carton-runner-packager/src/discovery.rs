//! This module implements runner discovery
//! See `docs/specification/runner.md` for more details

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use walkdir::WalkDir;

#[derive(Serialize, Deserialize)]
pub struct Config {
    /// Should be 1
    pub version: u64,

    /// A list of RunnerInfo structs
    pub runner: Vec<RunnerInfo>,
}

/// See `docs/specification/runner.md` for more details
#[derive(Serialize, Deserialize, Clone)]
pub struct RunnerInfo {
    pub runner_name: String,
    pub framework_version: semver::Version,
    pub runner_compat_version: u64,
    pub runner_interface_version: u64,
    pub runner_release_date: DateTime<Utc>,

    // Can be relative to the parsed file
    pub runner_path: String,

    // A target triple
    pub platform: String,
}

pub(crate) fn get_runner_dir() -> String {
    std::env::var("CARTON_RUNNER_DIR").unwrap_or("/usr/local/carton_runners".to_string())
}

#[derive(Debug, Error)]
enum DiscoveryError {
    #[error("IO error: {0}")]
    IOError(#[from] std::io::Error),

    #[error("Error parsing runner metadata: {0}")]
    ConfigParsingError(#[from] toml::de::Error),
}

/// Discover all installed runners
pub async fn discover_runners() -> Vec<RunnerInfo> {
    let runner_base_dir = get_runner_dir();

    // Find runner.toml files
    let mut runner_tomls = Vec::new();

    // Unfortunately not async
    let mut it = WalkDir::new(runner_base_dir).follow_links(true).into_iter();

    loop {
        let entry = match it.next() {
            None => break,
            Some(Err(_)) => continue,
            Some(Ok(entry)) => entry,
        };

        if entry.depth() > 0
            && entry.file_type().is_dir()
            && entry.file_name().to_str().unwrap().starts_with(".tmp")
        {
            // Ignore directories that start with ".tmp" as these are in-progress extractions
            it.skip_current_dir()
        }

        if entry.file_name() == "runner.toml" {
            runner_tomls.push(entry.into_path());

            // Skip the rest of this dir
            it.skip_current_dir()
        }
    }

    // Read all the configs
    let futs = runner_tomls.into_iter().map(|path| async move {
        let data = tokio::fs::read(&path).await?;
        let mut config: Config = toml::from_slice(&data)?;

        // This is safe because the last component is "runner.toml"
        let parent = path.parent().unwrap();

        // Since runner_path can be relative, let's turn it into an absolute one
        for runner in &mut config.runner {
            // (join handles absolute paths)
            runner.runner_path = parent
                .join(&runner.runner_path)
                .to_str()
                .unwrap()
                .to_owned();
        }

        Ok::<_, DiscoveryError>(config)
    });

    // Join and flatten
    futures::future::join_all(futs)
        .await
        .into_iter()
        .filter_map(|item| item.ok()) // Ignore parse errors. TODO: log
        .flat_map(|config| config.runner)
        .collect()
}
