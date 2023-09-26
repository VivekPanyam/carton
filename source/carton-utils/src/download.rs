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

use futures::StreamExt;
use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;
use tokio::sync::mpsc;
use tokio_util::io::ReaderStream;

use crate::{
    archive::with_atomic_extraction,
    config::CONFIG,
    error::{DownloadError, Result},
};

lazy_static! {
    // TODO: for some reason, if we allow HTTP2, requests hang when making
    // multiple parallel requests (e.g. when loading a model)
    // This is likely a bug within reqwest or something it uses under the hood
    static ref CLIENT: reqwest::Client = reqwest::ClientBuilder::new()
        .http1_only()
        .use_rustls_tls()
        .build()
        .unwrap();
}

/// Download a file with progress updates
/// Either download to a file or get a stream of chunks as the file is being downloaded (or both)
pub async fn uncached_download<P: AsRef<Path>>(
    url: &str,
    sha256: &str,
    download_path: Option<P>,
    chunk_stream: Option<mpsc::Sender<bytes::Bytes>>,
    mut on_content_len: impl FnMut(/* total */ Option<u64>),
    mut progress_update: impl FnMut(/* downloaded */ u64),
) -> Result<()> {
    // Create the file if necessary (can't use map because of the await)
    let mut outfile = match download_path {
        Some(download_path) => Some(tokio::fs::File::create(download_path).await.unwrap()),
        None => None,
    };

    // Download and copy to the target file while computing the sha256
    let mut hasher = Sha256::new();
    let mut res = CLIENT.get(url).send().await?;

    if !res.status().is_success() {
        // TODO: return an error instead of panic
        panic!("Error fetching URL {}: {}", url, res.status());
    }

    on_content_len(res.content_length());
    let mut downloaded = 0;

    while let Some(chunk) = res.chunk().await? {
        // Compute hash in a blocking task
        let b = chunk.clone();
        let jh1 = tokio::task::spawn_blocking(move || hasher.chain_update(&b));

        // Send the chunk out on the stream if we have one
        if let Some(cs) = chunk_stream.as_ref() {
            cs.send(chunk.clone()).await.unwrap();
        }

        // Copy to disk if we need to
        if let Some(outfile) = outfile.as_mut() {
            tokio::io::copy(&mut chunk.as_ref(), outfile).await.unwrap();
        }

        hasher = jh1.await.unwrap();
        downloaded += chunk.len() as u64;
        progress_update(downloaded);
    }

    // Make sure the sha256 matches the expected value
    let actual_sha256 = format!("{:x}", hasher.finalize());
    if sha256 != actual_sha256 {
        return Err(DownloadError::Sha256Mismatch {
            actual: actual_sha256,
            expected: sha256.into(),
        });
    }

    Ok(())
}

#[derive(Serialize, Deserialize)]
struct InfoJson {
    url: String,
}

/// Download a file with progress updates
/// Either download to a file or get a stream of chunks as the file is being downloaded (or both)
pub async fn cached_download<P: AsRef<Path>>(
    url: &str,
    sha256: &str,
    download_path: Option<P>,
    mut chunk_stream: Option<mpsc::Sender<bytes::Bytes>>,
    on_content_len: impl FnMut(/* total */ Option<u64>),
    progress_update: impl FnMut(/* downloaded */ u64),
) -> Result<()> {
    // Create the cache dir if necessary
    let files_cache_dir = CONFIG.cache_dir.join("files");
    tokio::fs::create_dir_all(&files_cache_dir).await.unwrap();

    // Download if necessary
    // This is a noop if the target exists already
    with_atomic_extraction(
        &files_cache_dir.join(sha256),
        &mut chunk_stream,
        |download_dir, chunk_stream| async move {
            // Create the download dir
            tokio::fs::create_dir(&download_dir).await.unwrap();

            // Download
            uncached_download(
                url,
                sha256,
                Some(download_dir.join("file")),
                chunk_stream.take(),
                on_content_len,
                progress_update,
            )
            .await
            .unwrap();

            // Write the info.json file
            let info = InfoJson {
                url: url.to_owned(),
            };
            let serialized = serde_json::to_string_pretty(&info).unwrap();
            tokio::fs::write(download_dir.join("info.json"), serialized)
                .await
                .unwrap();
        },
    )
    .await;

    // We now have the file in the cache.
    // Copy it to our target if we have one
    let cached_path = files_cache_dir.join(sha256).join("file");
    if let Some(download_path) = download_path {
        tokio::fs::copy(&cached_path, download_path).await.unwrap();
    }

    // TODO: we can do this and the above in parallel
    // If we didn't download, we need to load the file and send chunks to the chunk stream
    // We can check this by seeing if `chunk_stream` is None (because of the `take` in the closure above)
    if let Some(chunk_stream) = chunk_stream {
        let f = tokio::fs::File::open(cached_path).await.unwrap();
        let mut stream = ReaderStream::with_capacity(f, 1_000_000);
        while let Some(chunk) = stream.next().await {
            chunk_stream.send(chunk.unwrap()).await.unwrap();
        }
    }

    Ok(())
}
