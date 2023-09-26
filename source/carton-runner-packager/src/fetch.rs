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

use crate::{
    discovery::{
        get_matching_installed_runner, get_matching_runner, FilterableAsRunner,
        RunnerFilterConstraints, RunnerInfo,
    },
    install, DownloadInfo,
};
use dashmap::DashMap;
use lazy_static::lazy_static;
use tokio::sync::OnceCell;

lazy_static! {
    static ref FETCH_CACHE: DashMap<String, OnceCell<Vec<DownloadInfo>>> = DashMap::new();
    static ref CLIENT: reqwest::Client = reqwest::Client::new();
}

pub struct RunnerInstallConstraints {
    pub id: Option<String>,
    pub filters: RunnerFilterConstraints,
}

async fn fetch_runners(index_url: &str) -> Vec<DownloadInfo> {
    FETCH_CACHE
        .entry(index_url.to_owned())
        .or_insert_with(|| OnceCell::new())
        .downgrade()
        .get_or_init(|| async {
            let res = CLIENT
                .get(index_url)
                .send()
                .await
                .unwrap()
                .bytes()
                .await
                .unwrap();
            let out: Vec<DownloadInfo> = serde_json::from_slice(&res).unwrap();
            out
        })
        .await
        .clone()
}

/// Get an installed runner that matches the constraints or install one
/// If `upgrade` is set, don't check existing runners first and attempt to install a newer one
pub async fn get_or_install_runner(
    index_url: &str,
    constraints: &RunnerInstallConstraints,
    upgrade: bool,
) -> Result<RunnerInfo, &'static str> {
    if !upgrade {
        // Check installed runners
        if let Some(info) =
            get_matching_installed_runner(&constraints.filters, &constraints.id).await
        {
            return Ok(info);
        }
    }

    if let Some(id) = &constraints.id {
        // Fetch a runner with the specified ID
        let to_download = fetch_runners(index_url)
            .await
            .into_iter()
            .find(|r| &r.id == id)
            .ok_or("No installable runner found matching the requested ID")?;

        install(to_download, false).await;
    } else {
        // Install
        let runners = fetch_runners(index_url).await;
        let to_download = get_matching_runner(runners, &constraints.filters)
            .await
            .ok_or("No local or installable runners found matching requirements.")?;

        install(to_download, false).await;
    }

    // Try discovery again
    get_matching_installed_runner(&constraints.filters, &constraints.id)
            .await
            .ok_or("We just installed a matching runner, but none found. Please file an issue on GitHub if you get this error.")
}

impl FilterableAsRunner for DownloadInfo {
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

    fn runner_release_date(&self) -> &chrono::DateTime<chrono::Utc> {
        &self.runner_release_date
    }
}
