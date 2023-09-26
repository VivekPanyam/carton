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

//! This module implements runner discovery
//! See `docs/specification/runner.md` for more details

use std::path::PathBuf;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use walkdir::WalkDir;

#[derive(Serialize, Deserialize)]
pub(crate) struct Config {
    /// Should be 1
    pub version: u64,

    /// The ID this runner was installed as. This is set by the installation code
    pub installation_id: Option<String>,

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

pub struct RunnerFilterConstraints {
    pub runner_name: Option<String>,
    pub framework_version_range: Option<semver::VersionReq>,
    pub runner_compat_version: Option<u64>,
    pub max_runner_interface_version: u64,
    pub platform: String,
}

pub(crate) fn get_runner_dir() -> &'static PathBuf {
    &carton_utils::config::CONFIG.runner_dir
}

#[derive(Debug, Error)]
enum DiscoveryError {
    #[error("IO error: {0}")]
    IOError(#[from] std::io::Error),

    #[error("Error parsing runner metadata: {0}")]
    ConfigParsingError(#[from] toml::de::Error),
}

/// Discover all installed runners
pub async fn discover_runners(installation_id_filter: &Option<String>) -> Vec<RunnerInfo> {
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
        .filter_map(|item| match item {
            Ok(config) => {
                if installation_id_filter.is_some() {
                    if &config.installation_id != installation_id_filter {
                        return None;
                    }
                }

                return Some(config);
            }
            Err(_) => {
                None // Ignore parse errors. TODO: log
            }
        })
        .flat_map(|config| config.runner)
        .collect()
}

/// Get an installed runner that matches the constraints (or None)
pub async fn get_matching_installed_runner(
    constraints: &RunnerFilterConstraints,
    installation_id_filter: &Option<String>,
) -> Option<RunnerInfo> {
    let local_runners = discover_runners(installation_id_filter).await;

    get_matching_runner(local_runners, constraints).await
}

pub(crate) trait FilterableAsRunner {
    fn runner_name(&self) -> &str;
    fn framework_version(&self) -> &semver::Version;
    fn runner_compat_version(&self) -> u64;
    fn runner_interface_version(&self) -> u64;
    fn platform(&self) -> &str;
    fn runner_release_date(&self) -> &DateTime<Utc>;
}

impl FilterableAsRunner for RunnerInfo {
    fn runner_name(&self) -> &str {
        &self.runner_name
    }

    fn framework_version(&self) -> &semver::Version {
        &self.framework_version
    }

    fn runner_compat_version(&self) -> u64 {
        self.runner_compat_version
    }

    fn runner_interface_version(&self) -> u64 {
        self.runner_interface_version
    }

    fn platform(&self) -> &str {
        &self.platform
    }

    fn runner_release_date(&self) -> &DateTime<Utc> {
        &self.runner_release_date
    }
}

/// Get an installed runner that matches the constraints (or None)
pub(crate) async fn get_matching_runner<T>(
    runners: impl IntoIterator<Item = T>,
    constraints: &RunnerFilterConstraints,
) -> Option<T>
where
    T: FilterableAsRunner,
{
    // Filter the runners to ones that match our requirements
    runners
        .into_iter()
        .filter_map(|runner| {
            // The runner name must be the same as the model we're trying to load
            if let Some(runner_name) = &constraints.runner_name {
                if runner_name != runner.runner_name() {
                    return None;
                }
            }

            // The runner's framework_version must satisfy the model's required range
            if let Some(framework_version_range) = &constraints.framework_version_range {
                if !framework_version_range.matches(runner.framework_version()) {
                    return None;
                }
            }

            // The runner compat version must be the same as the model we're trying to load
            // (this is kind of like a version for the `model` directory)
            // If an expected runner_compat_version was specified, check if it matches
            if let Some(runner_compat_version) = constraints.runner_compat_version {
                if runner_compat_version != runner.runner_compat_version() {
                    return None;
                }
            }

            // We need to be able to start the runner so its platform must match the requirements
            if runner.platform() != constraints.platform {
                return None;
            }

            // Finally, we must be able to communicate with the runner (so its interface
            // version should be one we support)
            if runner.runner_interface_version() > constraints.max_runner_interface_version {
                return None;
            }

            Some(runner)
        })
        // Pick the newest one that matches the requirements
        .max_by_key(|item| item.runner_release_date().clone())
}
