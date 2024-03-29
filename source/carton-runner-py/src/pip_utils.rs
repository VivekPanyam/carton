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

use carton_runner_interface::slowlog::slowlog;
use serde::Deserialize;
use tokio::{process::Command, sync::OnceCell};

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

async fn ensure_has_pip() {
    static PIP_ONCE: tokio::sync::OnceCell<()> = OnceCell::const_new();

    PIP_ONCE.get_or_init(|| async {
        // Make sure we have 23.0
        install_wheel_and_make_available(
            "https://files.pythonhosted.org/packages/ab/43/508c403c38eeaa5fc86516eb13bb470ce77601b6d2bbcdb16e26328d0a15/pip-23.0-py3-none-any.whl",
            "b5f88adff801f5ef052bcdef3daa31b55eb67b0fccd6d0106c206fa248e0463c"
        ).await;

        // Make sure we have the `wheel` package
        install_wheel_and_make_available(
            "https://files.pythonhosted.org/packages/61/86/cc8d1ff2ca31a312a25a708c891cf9facbad4eae493b3872638db6785eb5/wheel-0.40.0-py3-none-any.whl",
            "d236b20e7cb522daf2390fa84c55eea81c5c30190f90f29ae2ca1ad8355bf247"
        ).await;
    }).await;
}

/// Effectively run
/// `python3 -m pip install --dry-run --ignore-installed --report {output_file} -r {requirements_file_path}`
/// and load the output
pub(crate) async fn get_pip_deps_report(requirements_file_contents: String) -> PipReport {
    // Make sure we have pip 23.0
    ensure_has_pip().await;

    // Create a file for the dependencies report
    let tempdir = tempfile::tempdir().unwrap();
    let output_file_path = tempdir.path().join("report.json");
    let requirements_file_path = tempdir.path().join("requirements.txt");

    // Write the requirements text contents to a file
    tokio::fs::write(&requirements_file_path, requirements_file_contents)
        .await
        .unwrap();

    let logs_tmp_dir = std::env::temp_dir().join("carton_logs");
    tokio::fs::create_dir_all(&logs_tmp_dir).await.unwrap();

    let log_dir = tempfile::tempdir_in(logs_tmp_dir).unwrap();
    log::info!(target: "slowlog", "Finding transitive dependencies using `pip install --report`. This may take a while. See the `pip` logs in {:#?}", log_dir.path());

    let mut sl = slowlog("`pip install --report`", 5)
        .await
        .without_progress();

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

    sl.done();

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
    use tokio::process::Command;

    use crate::{
        pip_utils::{ensure_has_pip, get_pip_deps_report},
        python_utils::get_executable_path,
    };

    #[tokio::test]
    async fn test_get_lightgbm_deps() {
        let requirements_file_contents = "lightgbm==3.3.5".to_owned();

        let report = get_pip_deps_report(requirements_file_contents).await;

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

    /// Ensure that the correct version of pip is available in subprocesses
    #[tokio::test]
    async fn test_pip_subprocess_version() {
        ensure_has_pip().await;

        let output = Command::new(get_executable_path().unwrap().as_str())
            .args(["-c", "import pip; print(pip.__version__)"])
            .output()
            .await
            .unwrap()
            .stdout;

        let p = String::from_utf8(output).unwrap();
        assert_eq!("23.0", p.trim());
    }

    #[tokio::test]
    async fn test_wheel_subprocess() {
        ensure_has_pip().await;

        let output = Command::new(get_executable_path().unwrap().as_str())
            .args(["-c", "import wheel; print(wheel.__version__)"])
            .output()
            .await
            .unwrap()
            .stdout;

        let p = String::from_utf8(output).unwrap();
        assert_eq!("0.40.0", p.trim());
    }
}
