use lazy_static::lazy_static;
use sha2::{Digest, Sha256};
use std::path::Path;

use crate::error::{DownloadError, Result};

lazy_static! {
    static ref CLIENT: reqwest::Client = reqwest::Client::new();
}

/// Download a file with progress updates
pub async fn uncached_download(
    url: &str,
    sha256: &str,
    download_path: &Path,
    mut on_content_len: impl FnMut(/* total */ Option<u64>),
    mut progress_update: impl FnMut(/* downloaded */ u64),
) -> Result<()> {
    // Create the file
    let mut outfile = tokio::fs::File::create(&download_path).await.unwrap();

    // Download and copy to the target file while computing the sha256
    let mut hasher = Sha256::new();
    let mut res = CLIENT.get(url).send().await.unwrap();

    on_content_len(res.content_length());
    let mut downloaded = 0;

    while let Some(chunk) = res.chunk().await.unwrap() {
        // TODO: see if we should offload this to a blocking thread
        hasher.update(&chunk);
        tokio::io::copy(&mut chunk.as_ref(), &mut outfile)
            .await
            .unwrap();
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
