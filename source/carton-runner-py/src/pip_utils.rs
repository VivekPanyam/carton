use std::{path::PathBuf, sync::atomic::AtomicBool};

use serde::Deserialize;
use tokio::process::Command;

use crate::{python_utils::get_executable_path, wheel::install_wheel_and_make_available};

#[derive(Debug, Deserialize)]
pub(crate) struct PipReport {
    pub install: Vec<PipInstallInfo>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct PipInstallInfo {
    pub download_info: PipDownloadInfo,
}

#[derive(Debug, Deserialize)]
pub(crate) struct PipDownloadInfo {
    pub url: String,
    pub archive_info: PipArchiveInfo,
}

#[derive(Debug, Deserialize)]
pub(crate) struct PipArchiveInfo {
    pub hashes: PipHashes,
}

#[derive(Debug, Deserialize)]
pub(crate) struct PipHashes {
    pub sha256: String,
}

static DID_INIT_PIP: AtomicBool = AtomicBool::new(false);
async fn ensure_has_pip() {
    if let Ok(_) = DID_INIT_PIP.compare_exchange(
        false,
        true,
        std::sync::atomic::Ordering::Relaxed,
        std::sync::atomic::Ordering::Relaxed,
    ) {
        // Make sure we have 23.0
        install_wheel_and_make_available(
            "https://files.pythonhosted.org/packages/ab/43/508c403c38eeaa5fc86516eb13bb470ce77601b6d2bbcdb16e26328d0a15/pip-23.0-py3-none-any.whl",
            "b5f88adff801f5ef052bcdef3daa31b55eb67b0fccd6d0106c206fa248e0463c"
        ).await;
    }
}

/// Effectively run
/// `python3 -m pip install --dry-run --ignore-installed --report {output_file} -r {requirements_file_path}`
/// and load the output
pub(crate) async fn get_pip_deps_report(requirements_file_path: PathBuf) -> PipReport {
    // Make sure we have pip 23.0
    ensure_has_pip().await;

    // Create a file for the dependencies report
    let tempdir = tempfile::tempdir().unwrap();
    let output_file_path = tempdir.path().join("report.json");

    let log_dir = tempfile::tempdir().unwrap();
    log::info!(target: "slowlog", "Finding transitive dependencies using `pip install --report`. This may take a while. See the `pip` logs in {:#?}", log_dir);

    // Run pip in a new process to isolate it a little bit from our embedded interpreter
    let success = Command::new(get_executable_path().unwrap().as_str())
        .args([
            "-m",
            "pip",
            "-q",
            "install",
            "--dry-run",
            "--ignore-installed",
            "--report",
            output_file_path.to_str().unwrap(),
            "-r",
            requirements_file_path.to_str().unwrap(),
        ])
        .stdout(std::fs::File::create(log_dir.path().join("stdout.log")).unwrap())
        .stderr(std::fs::File::create(log_dir.path().join("stderr.log")).unwrap())
        .status()
        .await
        .expect("Failed to run pip")
        .success();

    if !success {
        // Don't delete the log dir if it failed
        panic!(
            "Getting dependencies with pip failed! See the logs in {:?}",
            log_dir.into_path()
        );
    }

    // Load the deps
    let locked_deps_data = tokio::fs::read(&output_file_path).await.unwrap();
    serde_json::from_slice(&locked_deps_data).unwrap()
}

#[cfg(test)]
mod tests {
    use crate::pip_utils::get_pip_deps_report;

    #[tokio::test]
    async fn test_get_torch_deps() {
        let tempdir = tempfile::tempdir().unwrap();

        let requirements_file_path = tempdir.path().join("requirements.txt");
        std::fs::write(&requirements_file_path, "lightgbm==3.3.5").unwrap();

        let report = get_pip_deps_report(requirements_file_path).await;

        assert!(report
            .install
            .iter()
            .any(|item| item.download_info.url.contains("scikit_learn")));
        assert!(report
            .install
            .iter()
            .any(|item| item.download_info.url.contains("numpy")));
        assert!(report
            .install
            .iter()
            .any(|item| item.download_info.url.contains("scipy")));
        assert!(report
            .install
            .iter()
            .any(|item| item.download_info.url.contains("joblib")));

        println!("{:#?}", report);
    }
}
