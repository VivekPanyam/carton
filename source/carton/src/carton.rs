use std::collections::HashMap;

use crate::error::Result;
use crate::load::discover_or_get_runner_and_launch;
use crate::{
    conversion_utils::convert_map,
    error::CartonError,
    info::CartonInfoWithExtras,
    load::Runner,
    types::{CartonInfo, LoadOpts, PackOpts, SealHandle, Tensor},
};

pub struct Carton {
    info: CartonInfoWithExtras,
    runner: Runner,

    /// An optional temp dir. This is used in `load_unpacked` to make sure the directory doesn't get
    /// deleted while we need it
    tempdir: Option<tempfile::TempDir>,
}

impl Carton {
    /// Load a carton given a url, path, etc and options
    pub async fn load(url_or_path: String, opts: LoadOpts) -> Result<Self> {
        let (info, runner) = crate::load::load(&url_or_path, opts).await?;

        Ok(Self {
            info,
            runner: runner.unwrap(),
            tempdir: None,
        })
    }

    /// Infer using a set of inputs.
    /// Consider using `seal` and `infer_with_handle` in pipelines
    pub async fn infer_with_inputs(
        &self,
        tensors: HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        match &self.runner {
            Runner::V1(runner) => Ok(convert_map(
                runner
                    .infer_with_inputs(convert_map(tensors))
                    .await
                    .map_err(|e| CartonError::ErrorFromRunner(e))?,
            )),
        }
    }

    /// "Seal" a set of inputs that will be used for inference.
    /// This lets carton start processing tensors (e.g. moving them to the correct devices) before
    /// actually running inference and can lead to more efficient pipelines.
    pub async fn seal(&self, tensors: HashMap<String, Tensor>) -> Result<SealHandle> {
        match &self.runner {
            Runner::V1(runner) => Ok(SealHandle(
                runner
                    .seal(convert_map(tensors))
                    .await
                    .map_err(|e| CartonError::ErrorFromRunner(e))?,
            )),
        }
    }

    /// Infer using a handle from `seal`.
    /// This approach can make inference pipelines more efficient vs just using `infer_with_inputs`
    pub async fn infer_with_handle(&self, handle: SealHandle) -> Result<HashMap<String, Tensor>> {
        match &self.runner {
            Runner::V1(runner) => Ok(convert_map(
                runner
                    .infer_with_handle(handle.0)
                    .await
                    .map_err(|e| CartonError::ErrorFromRunner(e))?,
            )),
        }
    }

    /// Pack a carton given a path and options. Returns the path of the output file
    #[cfg(not(target_family = "wasm"))]
    pub async fn pack(path: String, opts: PackOpts) -> Result<std::path::PathBuf> {
        use std::sync::Arc;

        // Launch a runner
        let (runner, runner_info) =
            discover_or_get_runner_and_launch(&opts, &crate::types::Device::CPU).await?;

        // Create a temp folder
        // SAFETY: this only needs to last until the end of this method so it's okay if we don't store `tempdir`
        let tempdir = tempfile::tempdir()?;

        // Convert it to a lunchbox path
        let temp_folder = lunchbox::path::Path::new(tempdir.path().to_str().unwrap());

        // Create a localfs
        let localfs = Arc::new(lunchbox::LocalFS::new().unwrap());

        // Ask the runner to pack the model
        let model_dir_path = match runner {
            Runner::V1(runner) => runner
                .pack(&localfs, path.as_ref(), temp_folder)
                .await
                .map_err(|e| CartonError::ErrorFromRunner(e))?,
        };

        // Save and package the model
        crate::format::v1::save(opts, model_dir_path.to_string().as_ref()).await
    }

    /// Pack a carton given a path and options
    /// Functionally equivalent to `pack` followed by `load`, but implemented in a more
    /// optimized way
    #[cfg(not(target_family = "wasm"))]
    pub async fn load_unpacked(
        path: String,
        mut pack_opts: PackOpts,
        load_opts: LoadOpts,
    ) -> Result<Self> {
        use std::sync::Arc;

        // Launch a runner
        let (runner, runner_info) =
            discover_or_get_runner_and_launch(&pack_opts, &crate::types::Device::CPU).await?;

        // Set the runner_compat_version if the user didn't
        pack_opts
            .runner
            .runner_compat_version
            .get_or_insert(runner_info.runner_compat_version);

        // Create a temp folder
        // SAFETY: this tempdir needs to last for the entire time this Carton exists
        let tempdir = tempfile::tempdir()?;

        // Convert it to a lunchbox path
        let temp_folder = lunchbox::path::Path::new(tempdir.path().to_str().unwrap());

        // Create a localfs
        let localfs = Arc::new(lunchbox::LocalFS::new().unwrap());

        // Ask the runner to pack the model
        let model_dir_path = match &runner {
            Runner::V1(runner) => runner
                .pack(&localfs, path.as_ref(), temp_folder)
                .await
                .map_err(|e| CartonError::ErrorFromRunner(e))?,
        };

        // Create a localfs with the new root
        // TODO: don't unwrap this one because it may fail if the runner returned an invalid path
        let localfs =
            Arc::new(lunchbox::LocalFS::with_base_dir(model_dir_path.to_string()).unwrap());

        // Ask the runner to load the model it just packed
        let info_with_extras = CartonInfoWithExtras {
            info: pack_opts,
            manifest_sha256: None,
        };

        // Merge in load opts
        let visible_device = load_opts.visible_device.clone();
        let info_with_extras = crate::load::merge_in_load_opts(info_with_extras, load_opts)?;

        // TODO: correctly merge `load_opts` into `info_with_extras`
        crate::load::load_model(&localfs, &runner, &info_with_extras, visible_device).await?;

        // Return a Carton
        Ok(Self {
            info: info_with_extras,
            runner,
            tempdir: Some(tempdir),
        })
    }

    /// Get info for the loaded model
    pub fn get_info(&self) -> &CartonInfo {
        &self.info.info
    }

    /// Get info for a model
    pub async fn get_model_info(url_or_path: String) -> Result<CartonInfo> {
        Ok(crate::load::get_carton_info(&url_or_path).await?.info)
    }

    /// Allocate a tensor
    pub async fn alloc_tensor() -> Result<Tensor> {
        todo!()
    }
}
