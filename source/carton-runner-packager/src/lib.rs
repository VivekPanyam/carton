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

use std::path::Path;

use async_zip::{write::ZipFileWriter, ZipEntryBuilder};
use carton_utils::{
    archive::{extract, with_atomic_extraction},
    download::cached_download,
};
use chrono::{DateTime, Utc};
use discovery::{get_runner_dir, Config, RunnerInfo};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use url::{ParseError, Url};

pub mod discovery;
pub mod fetch;

/// Package a runner along with additional list zip or tar files to download and unpack at installation time
/// `upload_runner` is a function that is given the data for a `runner.zip` file along with its sha256 and returns a url
pub async fn package(mut info: RunnerInfo, additional: Vec<DownloadItem>) -> RunnerPackage {
    // Create a zip file in memory
    let mut zip = Vec::new();
    let mut writer = ZipFileWriter::new(&mut zip);

    // Add the runner
    writer
        .write_entry_whole(
            ZipEntryBuilder::new("runner".to_string(), async_zip::Compression::Zstd)
                .attribute_compatibility(async_zip::AttributeCompatibility::Unix)
                .unix_permissions(0o755), // Everyone gets read + execute
            &tokio::fs::read(info.runner_path).await.unwrap(),
        )
        .await
        .unwrap();

    // Modify the runner path and create a runner.toml file
    info.runner_path = "./runner".into();
    let runner_toml = toml::to_string_pretty(&Config {
        version: 1,
        installation_id: None, // This is set at install time
        runner: vec![info.clone()],
    })
    .unwrap();

    // Add it to the zip
    writer
        .write_entry_whole(
            ZipEntryBuilder::new("runner.toml".to_string(), async_zip::Compression::Zstd)
                .attribute_compatibility(async_zip::AttributeCompatibility::Unix)
                .unix_permissions(0o644), // Everyone gets read,
            runner_toml.as_bytes(),
        )
        .await
        .unwrap();

    // Close the writer
    writer.close().await.unwrap();

    // This lets the user set the url or path and then get a `DownloadInfo` struct
    RunnerPackage::new(zip, "".into(), info, additional)
}

// TODO: add slowlog for long running downloads
/// Install the runner if it doesn't already exist
pub async fn install(info: DownloadInfo, allow_local_files: bool) {
    let runner_base_dir = get_runner_dir();

    // Create it if it doesn't exist
    tokio::fs::create_dir_all(&runner_base_dir).await.unwrap();

    // TODO: validate that this joined path is safe
    let runner_dir = runner_base_dir.join(&info.id);

    // Extract into a temp dir and then move to the actual location
    with_atomic_extraction(&runner_dir, (), |runner_dir, _| async move {
        let mut handles = Vec::new();
        for file in info.download_info {
            // If url is a local file, make sure allow_local_files is true
            if is_file_path(&file.url) && !allow_local_files {
                panic!(
                    "Tried to install runner from local file '{}', but `allow_local_files` was not set",
                    &file.url
                );
            }

            // TODO: validate that this joined path is safe
            let target_dir = runner_dir.join(&file.relative_path);

            // Spawn tasks to download and extract
            handles.push(tokio::spawn(async move {
                let tempdir = tempfile::tempdir().unwrap();
                let download_path = tempdir.path().join("download");

                // Check if we actually need to download anything
                let download_path = if is_file_path(&file.url) {
                    Path::new(&file.url)
                } else {
                    cached_download(&file.url, &file.sha256, Some(&download_path), None, |_| {}, |_| {})
                        .await
                        .unwrap();

                    &download_path
                };

                // Extract the file (zip, tar, tar.gz)
                extract(download_path, &target_dir).await;
            }))
        }

        // Wait for all the downloads and extractions
        for handle in handles {
            handle.await.unwrap();
        }

        // Modify the runner.toml file to set the installation id
        let runner_toml = runner_dir.join("runner.toml");
        let data = tokio::fs::read(&runner_toml).await.unwrap();
        let mut config: Config = toml::from_slice(&data).unwrap();
        config.installation_id = Some(info.id);
        tokio::fs::write(&runner_toml, toml::to_string_pretty(&config).unwrap()).await.unwrap();
    }).await;
}

// TODO: make this more robust
fn is_file_path(input: &str) -> bool {
    match Url::parse(input) {
        Ok(parsed) => match parsed.scheme() {
            "file" => true,
            _ => false,
        },
        // This is a file
        Err(ParseError::RelativeUrlWithoutBase) => true,
        Err(e) => panic!("{e:?}"),
    }
}

/// Represents a runner package
#[derive(Serialize, Deserialize)]
pub struct RunnerPackage {
    // The sha256 and relative path of the main runner zip file
    main_sha256: String,
    main_relative_path: String,

    /// Don't serialize or deserialize data
    #[serde(skip)]
    main_data: Vec<u8>,

    /// The generated package id
    package_id: String,
    info: RunnerInfo,
    additional: Vec<DownloadItem>,
}

impl RunnerPackage {
    fn new(
        data: Vec<u8>,
        relative_path: String,
        info: RunnerInfo,
        additional: Vec<DownloadItem>,
    ) -> Self {
        // Compute the sha256 of the main zip file
        let mut hasher = Sha256::new();
        hasher.update(&data);
        let zip_sha256 = format!("{:x}", hasher.finalize());

        // Compute the sha256 of the sha256s of the main zip file and all the items in `additional`
        // to generate a unique id for this runner package
        let mut hasher = Sha256::new();
        hasher.update(&zip_sha256);
        for item in &additional {
            hasher.update(&item.sha256);
        }

        let package_id = format!("{:x}", hasher.finalize());

        RunnerPackage {
            main_sha256: zip_sha256,
            main_relative_path: relative_path,
            main_data: data,
            info,
            additional,
            package_id,
        }
    }

    /// Returns the generated package id. This is derived from the hashes of all the contained
    /// data files.
    pub fn get_id(&self) -> &str {
        &self.package_id
    }

    /// Gets the data of the main runner zip file
    pub fn get_data(&self) -> &[u8] {
        &self.main_data
    }

    /// Returns the sha256 of `data`
    pub fn get_data_sha256(&self) -> &str {
        &self.main_sha256
    }

    pub fn get_download_info(mut self, url: String) -> DownloadInfo {
        // Insert the runner zip file at the beginning
        self.additional.insert(
            0,
            DownloadItem {
                url,
                sha256: self.main_sha256,
                relative_path: self.main_relative_path,
            },
        );

        // Create the download config
        DownloadInfo {
            runner_name: self.info.runner_name,
            id: self.package_id,
            framework_version: self.info.framework_version,
            runner_compat_version: self.info.runner_compat_version,
            runner_interface_version: self.info.runner_interface_version,
            runner_release_date: self.info.runner_release_date,
            download_info: self.additional,
            platform: self.info.platform,
        }
    }
}

/// Structs for the json blob representing a runner available for download
/// See `docs/specification/runner.md` for more details
#[derive(Serialize, Deserialize, Clone)]
pub struct DownloadInfo {
    pub runner_name: String,
    pub id: String,
    pub framework_version: semver::Version,
    pub runner_compat_version: u64,
    pub runner_interface_version: u64,
    pub runner_release_date: DateTime<Utc>,

    pub download_info: Vec<DownloadItem>,

    // A target triple
    pub platform: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct DownloadItem {
    pub url: String,
    pub sha256: String,
    pub relative_path: String,
}
