use std::path::PathBuf;

use carton_runner_interface::slowlog::slowlog;
use carton_utils::{
    archive::{extract_zip, with_atomic_extraction},
    download::uncached_download,
};
use lazy_static::lazy_static;

use crate::python_utils::add_to_sys_path;

lazy_static! {
    static ref PACKAGE_BASE_DIR: PathBuf = {
        let base = carton_utils::config::CONFIG
            .runner_data_dir
            .join("python/packages/");

        std::fs::create_dir_all(&base).unwrap();

        base
    };
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
        // TODO: we should probably also do some locking to avoid wasted parallel installations
        return target_dir;
    }

    // Create a temp dir
    let tempdir = tempfile::tempdir().unwrap();
    let download_path = tempdir.path().join("download");

    // Slow log on timeout
    let mut sl = slowlog(format!("Downloading file '{url}'"), 5).await;

    // Uncached because we don't want to store both compressed wheels and the unzipped versions below
    uncached_download(
        url,
        sha256,
        &download_path,
        |total| {
            if let Some(size) = total {
                sl.set_total(Some(bytesize::ByteSize(size)));
            }
        },
        |downloaded| {
            sl.set_progress(Some(bytesize::ByteSize(downloaded)));
        },
    )
    .await
    .unwrap();

    // Let the logging task know we're done downloading
    sl.done();

    let mut sl = slowlog(format!("Extracting file '{url}'"), 5)
        .await
        .without_progress();

    // Unzip
    with_atomic_extraction(&target_dir, |out_dir| extract_zip(download_path, out_dir)).await;

    sl.done();

    // Return the path to add to sys.path
    target_dir
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
