use std::collections::{HashMap, VecDeque};

use carton_runner_interface::types::RunnerOpt;
use carton_utils::archive::extract_zip;
use lunchbox::path::{LunchboxPathUtils, PathBuf};
use pyo3::prelude::*;

use crate::{
    env::EnvironmentMarkers, model::Model, packager::CartonLock, python_utils::add_to_sys_path,
    wheel::install_wheel_and_make_available,
};

pub(crate) async fn load<F>(
    fs: F,
    runner_opts: Option<HashMap<String, RunnerOpt>>,
) -> Result<Model, String>
where
    F: lunchbox::ReadableFileSystem + Sync,
    F::FileType: lunchbox::types::ReadableFile + Unpin + Send + 'static,
{
    if let Some(opts) = runner_opts {
        // Make sure that the entrypoint opts are correctly specified
        let entrypoint_package = opts
            .get("entrypoint_package")
            .ok_or("Expected runner options of `entrypoint_package` and `entrypoint_fn` to be set, but `entrypoint_package` was not set. This means the model was likely not packaged correctly or options were removed when loading the model.".to_owned())?;

        let entrypoint_fn = opts
            .get("entrypoint_fn")
            .ok_or("Expected runner options of `entrypoint_package` and `entrypoint_fn` to be set, but `entrypoint_fn` was not set. This means the model was likely not packaged correctly or options were removed when loading the model.".to_owned())?;

        let entrypoint_package = get_runner_opt_string(entrypoint_package).ok_or(
            "Expected the `entrypoint_package` option to be a string, but it was a different type.",
        )?;
        let entrypoint_fn = get_runner_opt_string(entrypoint_fn).ok_or(
            "Expected the `entrypoint_fn` option to be a string, but it was a different type.",
        )?;

        // Ensure we have a carton.lock file
        let lockfile_path = PathBuf::from(".carton/carton.lock");
        if !lockfile_path.exists(&fs).await {
            return Err("The model does not contain a .carton/carton.lock file (which should have been generated during packaging). Please use the official packager or file a github issue if you believe this error is not correct.".into());
        }

        // Check if we have a lockfile for the current environment
        let env = EnvironmentMarkers::get_current().unwrap();
        let lockfile: CartonLock =
            toml::from_slice(&fs.read(&lockfile_path).await.unwrap()).unwrap();

        let matching_entry = lockfile.entries.iter().find(|item| item.matches(&env));
        if matching_entry.is_none() {
            log::warn!("A lockfile matching the current environment was not found. It is highly recommended to generate a lockfile for all environments that you'll be running in. TODO: add link to docs. Attempting to fetch dependencies...");
            todo!();
        }

        // Create a temp folder to copy bundled wheels to (if any)
        let bundled_wheels = tempfile::tempdir().unwrap();

        // This folder will be added to sys.path
        let temp_packages = tempfile::tempdir().unwrap();

        // Handles for our parallel copies
        let mut handles = Vec::new();

        // Make sure we have all deps available
        let matching_entry = matching_entry.unwrap();
        for dep in &matching_entry.locked_deps {
            if let Some(url) = &dep.url {
                let url = url.clone();
                let sha256 = dep.sha256.clone();
                handles.push(tokio::spawn(async move {
                    // TODO: Make sure this is a PyPi URL
                    install_wheel_and_make_available(&url, &sha256).await;
                }));
            } else if let Some(bundled_whl_path) = &dep.bundled_whl_path {
                if PathBuf::from(bundled_whl_path).exists(&fs).await {
                    let mut f = fs.open(bundled_whl_path).await.unwrap();
                    let local_path = bundled_wheels.path().join(&dep.sha256);
                    let mut target = tokio::fs::File::create(&local_path).await.unwrap();
                    let temp_packages_dir = temp_packages.path().to_owned();

                    handles.push(tokio::spawn(async move {
                        // Copy the lunchbox file to a local one
                        tokio::io::copy(&mut f, &mut target).await.unwrap();

                        // Unzip to our temp packages dir for this model
                        extract_zip(&local_path, &temp_packages_dir).await;
                    }));
                } else {
                    return Err(format!("The .carton/carton.lock file references a file ({bundled_whl_path}) that does not exist. It is possible that the lockfile was added to version control but the referenced files were not. Please repackage the model and try again. TODO: link"));
                }
            }
        }

        // Wait until all the copies and downloads are done
        for handle in handles {
            handle.await.unwrap();
        }

        // Add the temp packages to sys.path
        add_to_sys_path(&vec![temp_packages.path()]).unwrap();

        // Copy the entire contents of the model to a tempdir
        let model_dir_outer = tempfile::tempdir().unwrap();
        let model_dir_path = model_dir_outer.path().join("_carton_model_module");

        // Handles for our parallel copies
        let mut handles = Vec::new();

        // TODO: handle loops, etc here if needed (e.g. once we support symlinks)
        let mut to_process = VecDeque::new();
        to_process.push_back(PathBuf::from("."));
        while let Some(path) = to_process.pop_front() {
            let mut dir = fs.read_dir(path).await.unwrap();
            while let Some(entry) = dir.next_entry().await.unwrap() {
                // TODO: entry.metadata() doesn't follow symlinks
                if entry.metadata().await.unwrap().is_dir() {
                    to_process.push_back(entry.path());
                } else {
                    let filepath = entry.path();
                    let target_path = filepath.to_path(&model_dir_path);
                    assert!(target_path.starts_with(&model_dir_path));

                    let mut f = fs.open(filepath).await.unwrap();
                    tokio::fs::create_dir_all(target_path.parent().unwrap())
                        .await
                        .unwrap();
                    let mut target = tokio::fs::File::create(target_path).await.unwrap();

                    handles.push(tokio::spawn(async move {
                        // Copy the lunchbox file to a local one
                        tokio::io::copy(&mut f, &mut target).await.unwrap();
                    }));
                }
            }
        }

        // Wait for all the copies to finish
        for handle in handles {
            handle.await.unwrap();
        }

        // Add model_dir_outer to sys.path
        add_to_sys_path(&vec![model_dir_outer.path()]).unwrap();

        // TODO: maybe we need to clear the importlib cache

        let module_name = model_dir_path.file_name().unwrap().to_str().unwrap();
        let module_name = format!("{module_name}.{entrypoint_package}");

        let model = Python::with_gil(|py| {
            // Import the module
            let module = PyModule::import(py, module_name.as_str()).unwrap();

            // Get the entrypoint and run it to get the "model" that we'll use for inference
            let model = module
                .getattr(entrypoint_fn.as_str())
                .unwrap()
                .call0()
                .unwrap();

            Model::new(model_dir_outer, temp_packages, model)
        });

        Ok(model)
    } else {
        Err("Expected runner options of `entrypoint_package` and `entrypoint_fn` to be set, but no options were set. This means the model was likely not packaged correctly or options were removed when loading the model.".into())
    }
}

fn get_runner_opt_string(opt: &RunnerOpt) -> Option<&String> {
    if let RunnerOpt::String(item) = opt {
        Some(item)
    } else {
        None
    }
}
