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

use std::collections::{HashMap, VecDeque};

use carton_runner_interface::{slowlog::slowlog, types::RunnerOpt};
use carton_utils::archive::extract_zip;
use lunchbox::path::{LunchboxPathUtils, PathBuf};
use path_clean::PathClean;
use pyo3::{prelude::*, types::PyDict};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, BufReader};
use tracing::Instrument;

use crate::{
    env::EnvironmentMarkers,
    model::{pyerr_to_string_with_traceback, Model},
    packager::CartonLock,
    python_utils::add_to_sys_path,
    wheel::install_wheel_and_make_available,
};

#[tracing::instrument(skip(fs))]
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

        // TODO: handle loops, etc here if needed (e.g. if we support directory symlinks)
        let mut to_process = VecDeque::new();
        to_process.push_back(PathBuf::from("."));
        while let Some(path) = to_process.pop_front() {
            let mut dir = fs.read_dir(path).await.unwrap();
            while let Some(entry) = dir.next_entry().await.unwrap() {
                // entry.metadata() doesn't follow symlinks
                let metadata = entry.metadata().await.unwrap();
                if metadata.is_dir() {
                    to_process.push_back(entry.path());
                } else if metadata.is_symlink() {
                    // Get the absolute path of the entry
                    let filepath = entry.path();
                    let target_path = filepath.to_path(&model_dir_path).clean();
                    assert!(target_path.starts_with(&model_dir_path));

                    // Get the symlink target
                    let symlink_target = fs
                        .read_link(&filepath)
                        .await
                        .unwrap()
                        .to_path(&model_dir_path)
                        .clean();
                    assert!(symlink_target.starts_with(&model_dir_path));

                    // Create the dirs containing the symlink
                    tokio::fs::create_dir_all(target_path.parent().unwrap())
                        .await
                        .unwrap();

                    // Create the symlink
                    // TODO: do we want to convert it to a relative symlink instead of an absolute one?
                    tokio::fs::symlink(symlink_target, target_path)
                        .await
                        .unwrap();
                } else {
                    // Get the absolute path of the entry
                    let filepath = entry.path();
                    let target_path = filepath.to_path(&model_dir_path).clean();
                    assert!(target_path.starts_with(&model_dir_path));

                    let f = fs.open(&filepath).await.unwrap();
                    tokio::fs::create_dir_all(target_path.parent().unwrap())
                        .await
                        .unwrap();
                    let mut target = tokio::fs::File::create(target_path).await.unwrap();

                    let mut sl = slowlog(format!("Loading file '{}'", &filepath), 5).await;

                    let len = fs.metadata(&filepath).await.unwrap().len();
                    sl.set_total(Some(bytesize::ByteSize(len)));

                    // 1mb buffer
                    let mut br = BufReader::with_capacity(1_000_000, f);

                    handles.push(tokio::spawn(
                        async move {
                            // Copy the lunchbox file to a local one
                            copy(&mut br, &mut target, 1_000_000, |progress| {
                                sl.set_progress(Some(bytesize::ByteSize(progress)))
                            })
                            .await
                            .unwrap();
                            // tokio::io::copy_buf(&mut br, &mut target).await.unwrap();

                            sl.done();
                        }
                        .instrument(tracing::trace_span!(
                            "copy_file_to_local",
                            filepath = filepath.as_str()
                        )),
                    ));
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

        #[cfg(not(target_os = "macos"))]
        tracing::info_span!("preload_cuda").in_scope(|| {
            // Because of https://github.com/pytorch/pytorch/issues/101314, we need to attempt to preload cuda deps
            Python::with_gil(|py| {
                PyModule::from_code(py, include_str!("preload_cuda.py"), "", "")
                    .unwrap()
                    .getattr("preload_cuda_deps")
                    .unwrap()
                    .call0()
                    .unwrap();
            });
        });

        let module_name = model_dir_path.file_name().unwrap().to_str().unwrap();
        let module_name = format!("{module_name}.{entrypoint_package}");

        // Change directory to the model dir
        std::env::set_current_dir(&model_dir_path).unwrap();

        let model = tracing::info_span!("run_entrypoint").in_scope(|| {
            Python::with_gil(|py| {
                // Import the module
                let module = PyModule::import(py, module_name.as_str()).unwrap();

                // Get all the custom options specified by the user (anything starting with `model.`)
                let kwargs = PyDict::new(py);
                for (key, val) in &opts {
                    if let Some(key) = key.strip_prefix("model.") {
                        kwargs
                            .set_item(
                                key,
                                match val {
                                    RunnerOpt::Integer(v) => v.into_py(py),
                                    RunnerOpt::Double(v) => v.into_py(py),
                                    RunnerOpt::String(v) => v.into_py(py),
                                    RunnerOpt::Boolean(v) => v.into_py(py),
                                },
                            )
                            .unwrap();
                    }
                }

                // Get the entrypoint and run it to get the "model" that we'll use for inference
                let model = module
                    .getattr(entrypoint_fn.as_str())
                    .unwrap()
                    .call((), Some(kwargs))
                    .map_err(pyerr_to_string_with_traceback)
                    .unwrap();

                Model::new(model_dir_outer, temp_packages, model)
            })
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

pub async fn copy<'a, R: AsyncRead + Unpin, W: AsyncWrite + Unpin>(
    r: &'a mut R,
    w: &'a mut W,
    buf_size: usize,
    mut progress_update: impl FnMut(/* downloaded */ u64),
) -> std::io::Result<()> {
    let mut downloaded = 0;

    let mut buf = vec![0; buf_size];

    loop {
        let num_bytes = r.read(&mut buf).await?;

        if num_bytes == 0 {
            return Ok(());
        }

        let mut data = &buf[0..num_bytes];
        tokio::io::copy_buf(&mut data, w).await?;
        downloaded += num_bytes as u64;
        progress_update(downloaded);
    }
}
