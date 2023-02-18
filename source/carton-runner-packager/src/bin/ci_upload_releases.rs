//! A binary that uploads nightly releases. Used in CI

use std::{cmp::Reverse, sync::Arc};

use base64::Engine;
use carton_runner_packager::{DownloadInfo, RunnerPackage};
use s3::{creds::Credentials, Bucket, Region};
use serde::{Deserialize, Serialize};

#[tokio::main]
async fn main() {
    let API_KEY = std::env::var("NIGHTLY_REPO_TOKEN").expect("NIGHTLY_REPO_TOKEN should be in env");

    // First, get the current config and its blob sha
    let client = reqwest::Client::new();
    let res = client
        .get("https://api.github.com/repos/vivekpanyam/carton-nightly/contents/v1/runners")
        .header("Authorization", format!("Bearer {API_KEY}"))
        .header("X-GitHub-Api-Version", "2022-11-28")
        .header("Accept", "application/vnd.github.object")
        .header("User-Agent", "Carton-Nightly-Build")
        .send()
        .await
        .unwrap();

    if res.status() != 200 {
        panic!(
            "Got non-200 status: {}. Contents: {}",
            res.status(),
            res.text().await.unwrap()
        );
    }

    // Parse the object response
    // https://docs.github.com/en/rest/repos/contents?apiVersion=2022-11-28
    #[derive(Deserialize, Debug)]
    struct GHResponse {
        content: String,
        sha: String,
    }

    let github_response: GHResponse = res.json().await.unwrap();
    if github_response.content == "" {
        panic!("The github response content was empty. This likely means the runners file is > 1mb. See the GH API docs for more details.")
    }

    // Decode the runners file
    let contents = base64::engine::general_purpose::STANDARD
        .decode(github_response.content.trim())
        .unwrap();
    let mut runners: Vec<DownloadInfo> = serde_json::from_slice(&contents).unwrap();

    // Get the bucket to upload to
    let bucket = Arc::new(
        Bucket::new(
            &std::env::var("CARTON_NIGHTLY_S3_BUCKET").unwrap(),
            Region::Custom {
                region: std::env::var("CARTON_NIGHTLY_S3_REGION").unwrap(),
                endpoint: std::env::var("CARTON_NIGHTLY_S3_ENDPOINT").unwrap(),
            },
            Credentials::new(
                Some(&std::env::var("CARTON_NIGHTLY_ACCESS_KEY_ID").unwrap()),
                Some(&std::env::var("CARTON_NIGHTLY_SECRET_ACCESS_KEY").unwrap()),
                None,
                None,
                None,
            )
            .unwrap(),
        )
        .unwrap(),
    );

    // Get all the artifact dirs in /tmp/artifacts
    let handles = std::fs::read_dir("/tmp/artifacts")
        .unwrap()
        .into_iter()
        .flat_map(|artifact_dir| {
            let artifact_dir = artifact_dir.unwrap();

            // Get all the packages in this dir
            std::fs::read_dir(&artifact_dir.path())
                .unwrap()
                .filter_map(move |item| {
                    if let Ok(item) = item {
                        if item.file_name().to_str().unwrap().ends_with(".json") {
                            // Read the config
                            let package: RunnerPackage =
                                serde_json::from_slice(&std::fs::read(item.path()).unwrap())
                                    .unwrap();

                            // Get the zipfile path
                            let zipfile_path = artifact_dir
                                .path()
                                .join(format!("{}.zip", package.get_data_sha256()));

                            return Some((zipfile_path, package));
                        }
                    }

                    None
                })
        })
        .map(|(zip_path, package)| {
            let bucket = bucket.clone();
            tokio::spawn(async move {
                let content = tokio::fs::read(zip_path).await.unwrap();

                // Upload the zip file
                bucket
                    .put_object(format!("/{}", package.get_data_sha256()), &content)
                    .await
                    .unwrap();

                // Get the download_info
                let url = format!(
                    "https://nightly-assets.carton.run/{}",
                    package.get_data_sha256()
                );

                package.get_download_info(url)
            })
        });

    // Wait for the uploads to complete and then insert the runners
    for handle in handles {
        runners.insert(0, handle.await.unwrap());
    }

    // Sort in descending order
    runners.sort_by_key(|item| Reverse(item.runner_release_date));

    // Serialize
    let new_contents = serde_json::to_string_pretty(&runners).unwrap();

    // Upload
    // https://docs.github.com/en/rest/repos/contents?apiVersion=2022-11-28
    #[derive(Serialize, Debug)]
    struct GHUpload {
        message: String,
        content: String,
        sha: String,
        committer: GHAuthor,
    }

    #[derive(Serialize, Debug)]
    struct GHAuthor {
        name: &'static str,
        email: &'static str,
    }

    let to_upload = GHUpload {
        message: "Update nightly runners".to_string(),
        content: base64::engine::general_purpose::STANDARD.encode(new_contents),

        // The previous file's sha
        sha: github_response.sha,
        committer: GHAuthor {
            name: "CartonBot",
            email: "bot@carton.run",
        },
    };

    let res = client
        .put("https://api.github.com/repos/vivekpanyam/carton-nightly/contents/v1/runners")
        .header("Authorization", format!("Bearer {API_KEY}"))
        .header("X-GitHub-Api-Version", "2022-11-28")
        .header("Accept", "application/vnd.github+json")
        .header("User-Agent", "Carton-Nightly-Build")
        .body(serde_json::to_string(&to_upload).unwrap())
        .send()
        .await
        .unwrap();

    if res.status() != 200 {
        panic!(
            "Got non-200 status when updating runners file: {}. Contents: {}",
            res.status(),
            res.text().await.unwrap()
        );
    }
}
