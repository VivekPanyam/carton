use std::collections::HashMap;

use crate::types::{CartonInfo, LoadOpts, PackOpts, SealHandle, Tensor};

pub struct Carton {}

impl Carton {
    /// Load a carton given a url, path, etc and options
    pub async fn load(url_or_path: String, opts: LoadOpts) -> Self {
        todo!()
    }

    /// Infer using a set of inputs.
    /// Consider using `seal` and `infer_with_handle` in pipelines
    pub async fn infer_with_inputs(
        &self,
        tensors: HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>, String> {
        // Seal
        let handle = self.seal(tensors).await?;

        // Run inference
        self.infer_with_handle(handle).await
    }

    /// "Seal" a set of inputs that will be used for inference.
    /// This lets carton start processing tensors (e.g. moving them to the correct devices) before
    /// actually running inference and can lead to more efficient pipelines.
    pub async fn seal(&self, tensors: HashMap<String, Tensor>) -> Result<SealHandle, String> {
        todo!()
    }

    /// Infer using a handle from `seal`.
    /// This approach can make inference pipelines more efficient vs just using `infer_with_inputs`
    pub async fn infer_with_handle(
        &self,
        handle: SealHandle,
    ) -> Result<HashMap<String, Tensor>, String> {
        todo!()
    }

    /// Pack a carton given a path and options
    pub async fn pack(path: String, opts: PackOpts) {}

    /// Pack a carton given a path and options
    /// Functionally equivalent to `pack` followed by `load`, but implemented in a more
    /// optimized way
    pub async fn load_unpacked(path: String, pack_opts: PackOpts, load_opts: LoadOpts) -> Self {
        todo!()
    }

    /// Get info for the loaded model
    pub fn get_info(&self) -> &CartonInfo {
        todo!()
    }

    /// Get info for a model
    pub async fn get_model_info(url_or_path: String) -> CartonInfo {
        todo!()
    }

    /// Allocate a tensor
    pub async fn alloc_tensor() -> Tensor {
        todo!()
    }
}
