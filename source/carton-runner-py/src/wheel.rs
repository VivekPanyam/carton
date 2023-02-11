use std::path::{Path, PathBuf};

use async_zip::read::fs::ZipFileReader;
use carton_runner_interface::slowlog::slowlog;
use lazy_static::lazy_static;
use path_clean::PathClean;
use pyo3::Python;
use sha2::{Digest, Sha256};

use crate::python_utils::add_to_sys_path;

lazy_static! {
    static ref PACKAGE_BASE_DIR: PathBuf = Python::with_gil(|py| {
        let info = py.version_info();

        let base = home::home_dir().unwrap().join(format!(
            ".carton/pythonpackages/py{}{}/",
            info.major, info.minor
        ));

        std::fs::create_dir_all(&base).unwrap();

        base
    });
    static ref CLIENT: reqwest::Client = reqwest::Client::new();
}

/// Installs a wheel (if not already installed) and adds it to `sys.path`
pub async fn install_wheel_and_make_available(url: &str, sha256: &str) {
    let path = install_wheel(url, sha256).await;
    add_to_sys_path(&vec![path]).unwrap();
}

/// Installs a wheel file (if not already installed) and returns the path to add to `sys.path`
///
/// See the wheel spec at https://packaging.python.org/en/latest/specifications/binary-distribution-format/
/// There's a bit more to it, but a basic install just unzips the file into the target directory
pub async fn install_wheel(url: &str, sha256: &str) -> PathBuf {
    let target_dir = PACKAGE_BASE_DIR.join(sha256);
    if target_dir.exists() {
        // This already exists
        // TODO: we should probably do some locking to avoid wasted parallel installations
        return target_dir;
    }

    // Create a temp dir
    let tempdir = tempfile::tempdir().unwrap();
    let download_path = tempdir.path().join("download");

    let mut outfile = tokio::fs::File::create(&download_path).await.unwrap();

    // Download and copy to the target file while computing the sha256
    let mut hasher = Sha256::new();
    let mut res = CLIENT.get(url).send().await.unwrap();

    // Slow log on timeout
    let mut sl = match res.content_length() {
        Some(size) => {
            slowlog(
                format!("Downloading file '{url}' ({})", bytesize::ByteSize(size)),
                5,
            )
            .await
        }
        None => slowlog(format!("Downloading file '{url}'"), 5).await,
    };

    while let Some(chunk) = res.chunk().await.unwrap() {
        // TODO: see if we should offload this to a blocking thread
        hasher.update(&chunk);
        tokio::io::copy(&mut chunk.as_ref(), &mut outfile)
            .await
            .unwrap();
    }

    // Let the logging task know we're done downloading
    sl.done();

    // Make sure the sha256 matches the expected value
    let actual_sha256 = format!("{:x}", hasher.finalize());

    // TODO: return an error instead of asserting
    assert_eq!(sha256, actual_sha256);

    let mut sl = slowlog(format!("Extracting file '{url}'"), 5).await;

    // Unzip
    let extraction_dir = tempdir.path().join("extraction");
    unzip_file(&download_path, &extraction_dir).await;

    sl.done();

    // Move to the target directory. This should be atomic so it won't break anything
    // if multiple installs happen at the same time.
    match tokio::fs::rename(extraction_dir, &target_dir).await {
        Err(e) if e.raw_os_error() == Some(libc::ENOTEMPTY) => {
            // This can happen if another installation created the target directory before we called
            // rename.
            // We don't need to do anything here
        }
        e => e.unwrap(),
    }

    // Return the path to add to sys.path
    target_dir
}

// Based on https://github.com/Majored/rs-async-zip/blob/main/examples/file_extraction.rs
/// Extracts everything from the ZIP archive to the output directory
pub(crate) async fn unzip_file(archive: &Path, out_dir: &Path) {
    let mut handles = Vec::new();
    let reader = ZipFileReader::new(archive)
        .await
        .expect("Failed to read zip file");
    for index in 0..reader.file().entries().len() {
        let entry = &reader.file().entries().get(index).unwrap().entry();

        // Normalize the file path
        let path = out_dir.join(entry.filename()).clean();

        // Ensure that path is within the base dir
        if !path.starts_with(out_dir) {
            panic!("Error: extracted file path does not start with the output dir")
        }

        // If the filename of the entry ends with '/', it is treated as a directory.
        // This is implemented by previous versions of this crate and the Python Standard Library.
        // https://docs.rs/async_zip/0.0.8/src/async_zip/read/mod.rs.html#63-65
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
            let mut writer = tokio::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .await
                .expect("Failed to create extracted file");

            // Spawn a task to extract
            let reader = reader.clone();
            handles.push(tokio::spawn(async move {
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

#[cfg(test)]
mod tests {
    use tokio::process::Command;

    use crate::python_utils::get_executable_path;

    use super::{install_wheel, install_wheel_and_make_available, PACKAGE_BASE_DIR};

    #[tokio::test]
    async fn test_install_pip() {
        let out = install_wheel(
            "https://files.pythonhosted.org/packages/ab/43/508c403c38eeaa5fc86516eb13bb470ce77601b6d2bbcdb16e26328d0a15/pip-23.0-py3-none-any.whl",
            "b5f88adff801f5ef052bcdef3daa31b55eb67b0fccd6d0106c206fa248e0463c"
        ).await;

        assert_eq!(
            out,
            PACKAGE_BASE_DIR
                .join("b5f88adff801f5ef052bcdef3daa31b55eb67b0fccd6d0106c206fa248e0463c")
        );
    }

    /// Ensure that wheels that we make available in this process are also available in subprocesses
    #[tokio::test]
    async fn test_install_pip_subprocess() {
        install_wheel_and_make_available(
            "https://files.pythonhosted.org/packages/ab/43/508c403c38eeaa5fc86516eb13bb470ce77601b6d2bbcdb16e26328d0a15/pip-23.0-py3-none-any.whl",
            "b5f88adff801f5ef052bcdef3daa31b55eb67b0fccd6d0106c206fa248e0463c"
        ).await;

        let output = Command::new(get_executable_path().unwrap().as_str())
            .args(["-c", "import sys; print(sys.path)"])
            .output()
            .await
            .unwrap()
            .stdout;

        let p = String::from_utf8(output).unwrap();
        assert!(p.contains("b5f88adff801f5ef052bcdef3daa31b55eb67b0fccd6d0106c206fa248e0463c"))
    }
}
