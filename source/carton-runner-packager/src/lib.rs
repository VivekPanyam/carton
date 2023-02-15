use std::{
    future::Future,
    path::{Path, PathBuf},
};

use async_zip::{write::ZipFileWriter, ZipEntryBuilder};
use carton_utils::{
    archive::{extract, with_atomic_extraction},
    download::uncached_download,
};
use chrono::{DateTime, Utc};
use discovery::{get_runner_dir, Config, RunnerInfo};
use serde::Serialize;
use sha2::{Digest, Sha256};
use url::{ParseError, Url};

pub mod discovery;

/// Package a runner along with additional list zip or tar files to download and unpack at installation time
/// `upload_runner` is a function that is given the data for a `runner.zip` file along with its sha256 and returns a url
pub async fn package<F, Fut>(
    mut info: RunnerInfo,
    mut additional: Vec<DownloadItem>,
    upload_runner: F,
) -> DownloadInfo
where
    F: FnOnce(Vec<u8>, String) -> Fut,
    Fut: Future<Output = String>,
{
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

    // Compute the sha256 of the zip file
    let mut hasher = Sha256::new();
    hasher.update(&zip);
    let zip_sha256 = format!("{:x}", hasher.finalize());

    // Upload the runner and get the url
    // TODO: cloning the sha256 makes lifetimes for the closure simpler. Figure out if there's a way to do this more cleanly
    let url = upload_runner(zip, zip_sha256.clone()).await;

    // Insert the runner zip file at the beginning
    additional.insert(
        0,
        DownloadItem {
            url,
            sha256: zip_sha256,
            relative_path: "".into(),
        },
    );

    // Compute the sha256 of the sha256s of all the items in `additional`
    // to generate a unique id for this runner package
    let mut hasher = Sha256::new();
    for item in &additional {
        hasher.update(&item.sha256);
    }

    let id = format!("{:x}", hasher.finalize());

    // Create the download config
    DownloadInfo {
        runner_name: info.runner_name,
        id,
        framework_version: info.framework_version,
        runner_compat_version: info.runner_compat_version,
        runner_interface_version: info.runner_interface_version,
        runner_release_date: info.runner_release_date,
        download_info: additional,
        platform: info.platform,
    }
}

// TODO: add slowlog for long running downloads
/// Install the runner if it doesn't already exist
pub async fn install(info: DownloadInfo, allow_local_files: bool) {
    let runner_base_dir = PathBuf::from(get_runner_dir());

    // TODO: validate that this joined path is safe
    let runner_dir = runner_base_dir.join(&info.id);

    // Extract into a temp dir and then move to the actual location
    with_atomic_extraction(&runner_dir, |runner_dir| async move {
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
                    // Uncached download because there's probably not a significant overlap between these files
                    // and other files we'll be downloading
                    uncached_download(&file.url, &file.sha256, &download_path, |_| {}, |_| {})
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

/// Structs for the json blob representing a runner available for download
/// See `docs/specification/runner.md` for more details
#[derive(Serialize)]
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

#[derive(Serialize)]
pub struct DownloadItem {
    pub url: String,
    pub sha256: String,
    pub relative_path: String,
}
