use std::collections::HashMap;

use crate::error::Result;
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
}

impl Carton {
    /// Load a carton given a url, path, etc and options
    pub async fn load(url_or_path: String, opts: LoadOpts) -> Result<Self> {
        let (info, runner) = crate::load::load(&url_or_path, opts).await?;

        Ok(Self {
            info,
            runner: runner.unwrap(),
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

    /// Pack a carton given a path and options
    pub async fn pack(path: String, opts: PackOpts) -> Result<()> {
        todo!()
    }

    /// Pack a carton given a path and options
    /// Functionally equivalent to `pack` followed by `load`, but implemented in a more
    /// optimized way
    pub async fn load_unpacked(
        path: String,
        pack_opts: PackOpts,
        load_opts: LoadOpts,
    ) -> Result<Self> {
        todo!()
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
