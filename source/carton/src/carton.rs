use std::collections::HashMap;

use crate::{types::{CartonInfo, LoadOpts, PackOpts, SealHandle, Tensor}, load::Runner};
use crate::error::Result;

pub struct Carton {
    info: CartonInfo,
    runner: Runner,
}

impl Carton {
    /// Load a carton given a url, path, etc and options
    pub async fn load(url_or_path: String, opts: LoadOpts) -> Result<Self> {
        let (info, runner) = crate::load::load(&url_or_path, opts).await?;

        Ok(Self { info, runner: runner.unwrap() })
    }

    /// Infer using a set of inputs.
    /// Consider using `seal` and `infer_with_handle` in pipelines
    pub async fn infer_with_inputs(
        &self,
        tensors: HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        // Seal
        let handle = self.seal(tensors).await?;

        // Run inference
        self.infer_with_handle(handle).await
    }

    /// "Seal" a set of inputs that will be used for inference.
    /// This lets carton start processing tensors (e.g. moving them to the correct devices) before
    /// actually running inference and can lead to more efficient pipelines.
    pub async fn seal(&self, tensors: HashMap<String, Tensor>) -> Result<SealHandle> {
        todo!()
    }

    /// Infer using a handle from `seal`.
    /// This approach can make inference pipelines more efficient vs just using `infer_with_inputs`
    pub async fn infer_with_handle(
        &self,
        handle: SealHandle,
    ) -> Result<HashMap<String, Tensor>> {
        todo!()
    }

    /// Pack a carton given a path and options
    pub async fn pack(path: String, opts: PackOpts) -> Result<()> {
        todo!()
    }

    /// Pack a carton given a path and options
    /// Functionally equivalent to `pack` followed by `load`, but implemented in a more
    /// optimized way
    pub async fn load_unpacked(path: String, pack_opts: PackOpts, load_opts: LoadOpts) -> Result<Self> {
        todo!()
    }

    /// Get info for the loaded model
    pub fn get_info(&self) -> &CartonInfo {
        &self.info
    }

    /// Get info for a model
    pub async fn get_model_info(url_or_path: String) -> Result<CartonInfo> {
        crate::load::get_carton_info(&url_or_path).await
    }

    /// Allocate a tensor
    pub async fn alloc_tensor() -> Result<Tensor> {
        todo!()
    }
}
