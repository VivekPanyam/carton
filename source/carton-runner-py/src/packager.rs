use carton_runner_interface::slowlog::slowlog;
use lunchbox::path::LunchboxPathUtils;
use serde::{Deserialize, Serialize};
use tokio::process::Command;
use url::Url;

use sha2::{Digest, Sha256};

use crate::{
    env::EnvironmentMarkers,
    pip_utils::{get_pip_deps_report, PipInstallInfo},
    python_utils::get_executable_path,
};

/// A the structure of a carton.lock toml file
#[derive(Serialize, Deserialize, Debug, PartialEq, Default)]
pub struct CartonLock {
    // A hash of the original dependencies used to generate the locked dependencies
    // This helps us avoid regenerating lockfiles when we don't need to
    pub orig_deps_hash: String,

    #[serde(rename = "lockfile")]
    pub entries: Vec<LockfileEntry>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct LockfileEntry {
    // The environment to match
    #[serde(flatten)]
    pub required_environment: EnvironmentMarkers,

    // A list of deps (and transitive deps)
    pub locked_deps: Vec<LockedDep>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct LockedDep {
    /// At load time, if we have a package that matches the sha256, we'll add it to sys.path without doing anything else
    pub sha256: String,

    /// Otherwise, if `url` is not None, we'll fetch the URL, confirm that the hash matches and fail otherwise.
    /// If it does match, then we'll unpack it and add to sys.path
    pub url: Option<String>,

    /// If neither of the above two match, if bundled_whl_path is not None, we'll locally unpack it and add it to sys.path.
    pub bundled_whl_path: Option<String>,
}

impl LockfileEntry {
    pub fn matches(&self, env: &EnvironmentMarkers) -> bool {
        // A utility to check if a value matches a constraint
        let does_match_constraint = |constraint: &Option<String>, value: &Option<String>| -> bool {
            match constraint {
                // This is not constrained so it matches
                None => true,

                // Check if the constraint match the value via string equality
                Some(constraint) => constraint == value.as_ref().unwrap(),
            }
        };

        let reqs = &self.required_environment;

        does_match_constraint(&reqs.os_name, &env.os_name)
            && does_match_constraint(&reqs.sys_platform, &env.sys_platform)
            && does_match_constraint(&reqs.platform_machine, &env.platform_machine)
            && does_match_constraint(
                &reqs.platform_python_implementation,
                &env.platform_python_implementation,
            )
            && does_match_constraint(&reqs.platform_release, &env.platform_release)
            && does_match_constraint(&reqs.platform_system, &env.platform_system)
            && does_match_constraint(&reqs.python_version, &env.python_version)
            && does_match_constraint(&reqs.python_full_version, &env.python_full_version)
            && does_match_constraint(&reqs.implementation_name, &env.implementation_name)
            && does_match_constraint(&reqs.implementation_version, &env.implementation_version)
    }
}

/// Generates a lockfile in a python project based on the requirements.txt
/// Avoids unnecessarily regenerating
pub async fn update_or_generate_lockfile<F, P>(fs: &F, code_dir: P)
where
    F: lunchbox::WritableFileSystem + Sync,
    F::FileType: lunchbox::types::WritableFile + Unpin,
    P: AsRef<lunchbox::path::Path>,
{
    let code_dir = code_dir.as_ref();

    // Load the requirements.txt file
    let requirements_file_path = code_dir.join("requirements.txt");
    let requirements_file = fs.read(&requirements_file_path).await.unwrap();

    // Generate a hash of the requirements.txt
    let mut hasher = Sha256::new();
    hasher.update(requirements_file);

    // Create a lockfile struct
    let mut lockfile = CartonLock::default();

    // This enables an easy check to decide if we need to recompute the lockfile
    lockfile.orig_deps_hash = format!("{:x}", hasher.finalize());

    // Get the current environment
    let env = EnvironmentMarkers::get_current().unwrap();

    // Check if we already have a lockfile
    let lockfile_path = code_dir.join(".carton/carton.lock");
    if lockfile_path.exists(fs).await {
        // Load the file
        let old_lockfile: CartonLock =
            toml::from_slice(&fs.read(&lockfile_path).await.unwrap()).unwrap();

        if old_lockfile.orig_deps_hash != lockfile.orig_deps_hash {
            // If orig_deps_hash doesn't match the one we just generated, we have to start from scratch
        } else {
            // Get all entries matching the current environment.
            let has_matching_entries = old_lockfile.entries.iter().any(|item| item.matches(&env));

            // If we have any matching entries and orig_deps_hash matches the one we just generated, we don't need to do anything else
            if has_matching_entries {
                return;
            }

            // We can start with the old lockfile
            lockfile = old_lockfile;
        }
    }

    let locked_deps = get_pip_deps_report(fs, requirements_file_path).await;

    // Utils
    let is_pypi = |item: &PipInstallInfo| {
        Url::parse(&item.download_info.url).unwrap().host_str() == Some("files.pythonhosted.org")
    };
    let is_wheel = |item: &PipInstallInfo| {
        Url::parse(&item.download_info.url)
            .unwrap()
            .path()
            .ends_with(".whl")
    };

    // pypi wheels are stored as urls
    let mut deps: Vec<LockedDep> = locked_deps
        .install
        .iter()
        .filter(|item| is_wheel(item) && is_pypi(item))
        .map(|item| LockedDep {
            sha256: item.download_info.archive_info.hashes.sha256.clone(),
            url: Some(item.download_info.url.clone()),
            bundled_whl_path: None,
        })
        .collect();

    // Wheels other than pypi ones will be stored in the carton (including the wheels we're going to build from source)
    // .carton/bundled_wheels/{sha256}/{wheel_name}.whl
    let client = reqwest::Client::new();
    let other_wheels = locked_deps
        .install
        .iter()
        .filter(|item| is_wheel(item) && !is_pypi(item));

    for item in other_wheels {
        // Figure out where to download the file to
        let parsed = Url::parse(&item.download_info.url).unwrap();
        let fname = parsed.path_segments().unwrap().last().unwrap();
        let sha256 = &item.download_info.archive_info.hashes.sha256;

        log::info!(target: "slowlog", "Fetching and bundling non-pypi wheel: {:#?}", parsed);

        let mut sl = slowlog(format!("Downloading file '{}'", &item.download_info.url), 5)
            .await
            .without_progress();

        let relative_path = format!(".carton/bundled_wheels/{sha256}/{fname}");
        let bundled_path = code_dir.join(&relative_path);
        if !bundled_path.exists(fs).await {
            fs.create_dir_all(bundled_path.parent().unwrap())
                .await
                .unwrap();
            let mut outfile = fs.create(&bundled_path).await.unwrap();

            // Download and copy to the target file
            let mut res = client.get(&item.download_info.url).send().await.unwrap();
            while let Some(chunk) = res.chunk().await.unwrap() {
                tokio::io::copy(&mut chunk.as_ref(), &mut outfile)
                    .await
                    .unwrap();
            }
        }

        sl.done();

        deps.push(LockedDep {
            sha256: sha256.into(),
            url: None,
            bundled_whl_path: Some(relative_path),
        })
    }

    // Finally, source packages will be built into wheels and bundled into the carton.
    // TODO cache this step (sha256 of input to sha256 of output?)

    // Create a tempdir for the wheels we're building
    let tempdir = tempfile::tempdir().unwrap();

    let source_packages = locked_deps
        .install
        .iter()
        .filter(|item| !is_wheel(item))
        .map(|item| item.download_info.url.as_str());

    if source_packages.clone().count() != 0 {
        let logs_tmp_dir = std::env::temp_dir().join("carton_logs");
        tokio::fs::create_dir_all(&logs_tmp_dir).await.unwrap();

        let log_dir = tempfile::tempdir_in(logs_tmp_dir).unwrap();
        log::info!(target: "slowlog", "Building wheels for non-wheel dependencies using `pip wheel`. This may take a while. See the `pip` logs in {:#?}", log_dir.path());

        let mut sl = slowlog("`pip wheel`", 5).await.without_progress();

        // Run pip in a new process to isolate it a little bit from our embedded interpreter
        let build_success = Command::new(get_executable_path().unwrap().as_str())
            .args(
                [
                    "-m",
                    "pip",
                    "-q",
                    "wheel",
                    "--no-deps",
                    "--wheel-dir",
                    tempdir.path().to_str().unwrap(),
                ]
                .into_iter()
                .chain(source_packages),
            )
            .stdout(std::fs::File::create(log_dir.path().join("stdout.log")).unwrap())
            .stderr(std::fs::File::create(log_dir.path().join("stderr.log")).unwrap())
            .status()
            .await
            .expect("Failed to run pip")
            .success();

        sl.done();

        if !build_success {
            // Don't delete the log dir if it failed
            panic!(
                "Failed to build wheels for dependencies! See the logs in {:?}",
                log_dir.into_path()
            );
        }
    }

    // TODO: parallelize this by splitting into smaller tasks
    // Also avoid loading the data twice (once to copy, once to compute the hash)
    let mut paths = tokio::fs::read_dir(tempdir.path()).await.unwrap();
    while let Some(item) = paths.next_entry().await.unwrap() {
        // Compute the sha256
        let mut hasher = Sha256::new();
        let buf = tokio::fs::read(item.path()).await.unwrap();
        hasher.update(&buf);
        let sha256 = format!("{:x}", hasher.finalize());

        // println!("Built wheel from source {:#?}", item.path());

        // Bundle the fiile into the wheel
        let relative_path = format!(
            ".carton/bundled_wheels/{}/{}",
            sha256,
            item.file_name().to_str().unwrap()
        );
        let bundled_path = code_dir.join(&relative_path);
        fs.create_dir_all(bundled_path.parent().unwrap())
            .await
            .unwrap();
        let mut outfile = fs.create(&bundled_path).await.unwrap();
        tokio::io::copy(&mut buf.as_slice(), &mut outfile)
            .await
            .unwrap();

        deps.push(LockedDep {
            sha256,
            url: None,
            bundled_whl_path: Some(relative_path),
        });
    }

    // Add the output to our lockfile
    lockfile.entries.push(LockfileEntry {
        // We only condition the lockfile on a subset of all the environment markers
        required_environment: EnvironmentMarkers {
            os_name: env.os_name,
            sys_platform: env.sys_platform,
            platform_machine: env.platform_machine,
            platform_python_implementation: env.platform_python_implementation,
            python_version: env.python_version,
            ..Default::default()
        },
        locked_deps: deps,
    });

    // Serialize it to a carton.lock toml
    let header = "
# THIS FILE IS AUTOGENERATED BY CARTON AND SHOULD NOT BE MANUALLY EDITED
# If you add this lockfile to version control, you MUST also add the entire `.carton` folder

";
    let serialized = toml::to_string_pretty(&lockfile).unwrap();
    fs.create_dir_all(lockfile_path.parent().unwrap())
        .await
        .unwrap();
    fs.write(lockfile_path, header.to_string() + &serialized)
        .await
        .unwrap();
}

mod tests {

    use super::{update_or_generate_lockfile, CartonLock};

    #[tokio::test]
    async fn test_generate_lockfile() {
        let tempdir = tempfile::tempdir().unwrap();

        let requirements_file_path = tempdir.path().join("requirements.txt");
        std::fs::write(&requirements_file_path, "xgboost==1.7.3").unwrap();

        let fs = lunchbox::LocalFS::new().unwrap();
        update_or_generate_lockfile(&fs, tempdir.path().to_str().unwrap()).await;

        let lockfile: CartonLock = toml::from_slice(
            &tokio::fs::read(&tempdir.path().join(".carton/carton.lock"))
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(lockfile.entries.len(), 1);
        assert!(lockfile.entries[0].locked_deps.iter().any(|item| item
            .url
            .as_ref()
            .unwrap()
            .contains("xgboost")));
        assert!(lockfile.entries[0].locked_deps.iter().any(|item| item
            .url
            .as_ref()
            .unwrap()
            .contains("scipy")));
        assert!(lockfile.entries[0].locked_deps.iter().any(|item| item
            .url
            .as_ref()
            .unwrap()
            .contains("numpy")));
    }
}
