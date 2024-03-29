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

use flate2::read::GzDecoder;
use path_clean::PathClean;
use std::{
    future::Future,
    io::Read,
    path::{Path, PathBuf},
    sync::Arc,
};
use tokio::sync::Semaphore;

use async_zip::read::fs::ZipFileReader;

// Based on https://github.com/Majored/rs-async-zip/blob/main/examples/file_extraction.rs
/// Extracts a ZIP archive to the output directory
pub async fn extract_zip<P: AsRef<Path>>(archive: P, out_dir: P) {
    let out_dir = out_dir.as_ref();
    let mut handles = Vec::new();
    let reader = ZipFileReader::new(archive)
        .await
        .expect("Failed to read zip file");

    // We want to limit the number of open files
    // This should let 64 file extractions run concurrently
    let open_files_semaphore = Arc::new(Semaphore::new(64));

    for index in 0..reader.file().entries().len() {
        let entry = &reader.file().entries().get(index).unwrap().entry();

        // Normalize the file path
        let path = out_dir.join(entry.filename()).clean();

        // Ensure that path is within the base dir
        if !path.starts_with(out_dir) {
            panic!("Error: extracted file path does not start with the output dir")
        }

        // If the filename of the entry ends with '/', it is treated as a directory.
        // This is implemented by the Python Standard Library.
        // https://github.com/python/cpython/blob/820ef62833bd2d84a141adedd9a05998595d6b6d/Lib/zipfile.py#L528
        let entry_is_dir = entry.filename().ends_with('/');

        if entry_is_dir {
            // The directory may have been created if iteration is out of order.
            if !path.exists() {
                tokio::fs::create_dir_all(&path)
                    .await
                    .expect("Failed to create extracted directory");
            }
        } else {
            // Creates parent directories. They may not exist if iteration is out of order
            // or the archive does not contain directory entries.
            let parent = path
                .parent()
                .expect("A file entry should have parent directories");
            if !parent.is_dir() {
                tokio::fs::create_dir_all(parent)
                    .await
                    .expect("Failed to create parent directories");
            }

            // Spawn a task to extract
            let open_files_semaphore = open_files_semaphore.clone();
            let reader = reader.clone();
            let perms = entry.unix_permissions().unwrap() as u32;
            handles.push(tokio::spawn(async move {
                // Limit number of open files
                let _permit = open_files_semaphore.acquire().await.unwrap();

                let mut writer = tokio::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .mode(perms)
                    .open(&path)
                    .await
                    .expect("Failed to create extracted file");
                let mut entry_reader = reader.entry(index).await.expect("Failed to read ZipEntry");
                tokio::io::copy(&mut entry_reader, &mut writer)
                    .await
                    .expect("Failed to copy to extracted file");
            }));
        }
    }

    // Wait until all the files are extracted
    for handle in handles {
        handle.await.unwrap();
    }
}

/// Extracts a tar.gz archive to the output directory
pub async fn extract_tar_gz<P: Into<PathBuf>>(archive: P, out_dir: P) {
    let archive = archive.into();
    let out_dir = out_dir.into();
    tokio::task::spawn_blocking(move || {
        let gz = std::fs::File::open(archive).unwrap();
        let tar = GzDecoder::new(gz);
        let mut archive = tar::Archive::new(tar);
        archive.unpack(&out_dir).unwrap();
    })
    .await
    .unwrap();
}

/// Extracts a tar archive to the output directory
pub async fn extract_tar<P: Into<PathBuf>>(archive: P, out_dir: P) {
    let archive = archive.into();
    let out_dir = out_dir.into();
    tokio::task::spawn_blocking(move || {
        let tar = std::fs::File::open(archive).unwrap();
        let mut archive = tar::Archive::new(tar);
        archive.unpack(&out_dir).unwrap();
    })
    .await
    .unwrap();
}

/// Extract an archive (either zip, tar, or tar.gz)
pub async fn extract(archive: &Path, out_dir: &Path) {
    // TODO: don't use `expect` and return an error
    let kind = infer::get_from_path(archive)
        .expect("file is read successfully")
        .expect("file type is known");

    match kind.mime_type() {
        "application/zip" => extract_zip(archive, out_dir).await,
        "application/gzip" => {
            let gz = std::fs::File::open(archive).unwrap();
            let decoder = GzDecoder::new(gz);

            // We only need the first 261 bytes to tell if it's a tar file
            let mut buf = Vec::with_capacity(512);
            decoder.take(512).read_to_end(&mut buf).unwrap();
            if infer::archive::is_tar(&buf) {
                extract_tar_gz(&archive, &out_dir).await;
            } else {
                panic!("Got a gz file but it wasn't a tar.gz");
            }
        }
        "application/x-tar" => {
            extract_tar(&archive, &out_dir).await;
        }
        other => panic!("Got an unsupported archive type: {other}"),
    }
}

/// This calls the provided function `do_extract` with a temporary path to extract into and then moves that dir to `target_dir`
/// This should be atomic so it won't cause broken output if multiple extractions happen at the same time.
/// This temporary directory is created in `target_dir.parent()` (with a name that starts with `.tmp`). This is necessary because
/// we need to ensure that the temp dir and the target are on the same device to avoid EXDEV errors when renaming.
/// Note: if `target_dir` exists, this function doesn't do anything
pub async fn with_atomic_extraction<F, A, Fut>(target_dir: &Path, args: A, do_extract: F)
where
    F: FnOnce(PathBuf, A) -> Fut,
    Fut: Future<Output = ()>,
{
    if target_dir.exists() {
        // This already exists
        // TODO: we should probably also do some locking to avoid wasted parallel extractions
        return;
    }

    // Create a temp dir
    let tempdir = tempfile::Builder::new()
        .prefix(".tmp")
        .tempdir_in(target_dir.parent().unwrap())
        .unwrap();
    let extraction_dir = tempdir.path().join("extraction");

    // Extract
    do_extract(extraction_dir.clone(), args).await;

    // Move to the target directory. This should be atomic so it won't break anything
    // if multiple installs happen at the same time.
    match tokio::fs::rename(&extraction_dir, &target_dir).await {
        Err(e)
            if e.raw_os_error() == Some(libc::ENOTEMPTY)
                || e.raw_os_error() == Some(libc::EEXIST) =>
        {
            // See https://man7.org/linux/man-pages/man2/rename.2.html
            // This can happen if another installation created the target directory before we called
            // rename.
            // We don't need to do anything here
        }
        e => e.unwrap(),
    }
}
