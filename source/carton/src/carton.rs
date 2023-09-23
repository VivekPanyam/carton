use std::collections::HashMap;

use carton_macros::for_each_carton_type;

use crate::error::Result;
use crate::load::discover_or_get_runner_and_launch;
use crate::runner_interface::storage::RunnerStorage;
use crate::types::{DataType, GenericStorage, TensorStorage};
use crate::{
    conversion_utils::convert_map,
    error::CartonError,
    info::CartonInfoWithExtras,
    load::Runner,
    types::{LoadOpts, PackOpts, SealHandle, Tensor},
};

pub struct Carton {
    info: CartonInfoWithExtras<GenericStorage>,
    runner: Runner,

    /// An optional temp dir. This is used in `load_unpacked` to make sure the directory doesn't get
    /// deleted while we need it
    _tempdir: Option<tempfile::TempDir>,
}

impl Carton {
    /// Load a carton given a url, path, etc and options
    pub async fn load<P: AsRef<str>>(url_or_path: P, opts: LoadOpts) -> Result<Self> {
        let (info, runner) = crate::load::load(url_or_path.as_ref(), opts).await?;

        Ok(Self {
            info,
            runner: runner.unwrap(),
            _tempdir: None,
        })
    }

    /// Infer using a set of inputs.
    /// Consider using `seal` and `infer_with_handle` in pipelines
    pub async fn infer<I, S, T>(&self, tensors: I) -> Result<HashMap<String, Tensor<RunnerStorage>>>
    where
        I: IntoIterator<Item = (S, Tensor<T>)>,
        String: From<S>,
        T: TensorStorage,
    {
        match &self.runner {
            Runner::V1(runner) => Ok(convert_map(
                runner
                    .infer_with_inputs(
                        tensors
                            .into_iter()
                            .map(|(k, v)| (k.into(), v.into()))
                            .collect(),
                    )
                    .await
                    .map_err(|e| CartonError::ErrorFromRunner(e))?,
            )),
        }
    }

    /// "Seal" a set of inputs that will be used for inference.
    /// This lets carton start processing tensors (e.g. moving them to the correct devices) before
    /// actually running inference and can lead to more efficient pipelines.
    pub async fn seal<T>(&self, tensors: HashMap<String, Tensor<T>>) -> Result<SealHandle>
    where
        T: TensorStorage,
    {
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
    /// This approach can make inference pipelines more efficient vs just using `infer`
    pub async fn infer_with_handle(
        &self,
        handle: SealHandle,
    ) -> Result<HashMap<String, Tensor<RunnerStorage>>> {
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
    pub async fn pack<T, O, P: AsRef<str>>(path: P, opts: O) -> Result<std::path::PathBuf>
    where
        T: TensorStorage,
        O: Into<PackOpts<T>>,
    {
        use std::sync::Arc;

        let mut opts = opts.into();

        // Launch a runner
        let (runner, runner_info) =
            discover_or_get_runner_and_launch(&opts.info, &crate::types::Device::CPU).await?;

        // Set the runner_compat_version if the user didn't
        opts.info
            .runner
            .runner_compat_version
            .get_or_insert(runner_info.runner_compat_version);

        // Create a temp folder
        // SAFETY: this only needs to last until the end of this method so it's okay if we don't store `tempdir`
        let tempdir = tempfile::tempdir()?;

        // Convert it to a lunchbox path
        let temp_folder = lunchbox::path::Path::new(tempdir.path().to_str().unwrap());

        // Create a localfs
        let localfs = Arc::new(lunchbox::LocalFS::new().unwrap());

        // Ask the runner to pack the model
        log::trace!("Asking runner to pack...");
        let model_dir_path = match runner {
            Runner::V1(runner) => runner
                .pack(
                    &localfs,
                    lunchbox::path::Path::new(path.as_ref()),
                    temp_folder,
                )
                .await
                .map_err(|e| CartonError::ErrorFromRunner(e))?,
        };

        log::trace!("About to save the packed model...");

        // Save and package the model
        crate::format::v1::save(opts, model_dir_path.to_string().as_ref()).await
    }

    /// Pack a carton given a path and options
    /// Functionally equivalent to `pack` followed by `load`, but implemented in a more
    /// optimized way
    #[cfg(not(target_family = "wasm"))]
    pub async fn load_unpacked<T, O, P: AsRef<str>>(
        path: P,
        pack_opts: O,
        load_opts: LoadOpts,
    ) -> Result<Self>
    where
        T: TensorStorage + 'static,
        O: Into<PackOpts<T>>,
    {
        use std::sync::Arc;

        use crate::conversion_utils::ConvertInto;

        let mut pack_opts = pack_opts.into();

        // Launch a runner
        let (runner, runner_info) =
            discover_or_get_runner_and_launch(&pack_opts.info, &crate::types::Device::CPU).await?;

        // Set the runner_compat_version if the user didn't
        pack_opts
            .info
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
                .pack(
                    &localfs,
                    lunchbox::path::Path::new(path.as_ref()),
                    temp_folder,
                )
                .await
                .map_err(|e| CartonError::ErrorFromRunner(e))?,
        };

        // Create a localfs with the new root
        // TODO: don't unwrap this one because it may fail if the runner returned an invalid path
        let localfs = Arc::new(
            lunchbox::LocalFS::with_base_dir(model_dir_path.to_string())
                .await
                .unwrap(),
        );

        // Ask the runner to load the model it just packed
        let info_with_extras = CartonInfoWithExtras {
            info: pack_opts.info,
            manifest_sha256: None,
        };

        // Merge in load opts
        let visible_device = load_opts.visible_device.clone();
        let info_with_extras = crate::load::merge_in_load_opts(info_with_extras, load_opts)?;

        // TODO: correctly merge `load_opts` into `info_with_extras`
        crate::load::load_model(&localfs, &runner, &info_with_extras, visible_device).await?;

        // Return a Carton
        Ok(Self {
            info: info_with_extras.convert_into(),
            runner,
            _tempdir: Some(tempdir),
        })
    }

    /// Get info for the loaded model
    pub fn get_info(&self) -> &CartonInfoWithExtras<GenericStorage> {
        &self.info
    }

    /// Get info for a model
    pub async fn get_model_info<P: AsRef<str>>(
        url_or_path: P,
    ) -> Result<CartonInfoWithExtras<GenericStorage>> {
        crate::load::get_carton_info(url_or_path.as_ref()).await
    }

    /// Shrink a packed carton by storing links to files instead of the files themselves when possible.
    /// Takes a path to a packed carton along with a mapping from sha256 to a list of URLs
    /// Returns the path to another packed carton
    #[cfg(not(target_family = "wasm"))]
    pub async fn shrink(
        path: std::path::PathBuf,
        urls: HashMap<String, Vec<String>>,
    ) -> Result<std::path::PathBuf> {
        crate::format::v1::links::create_links(path, urls).await
    }

    /// Allocate a tensor
    pub async fn alloc_tensor(
        &self,
        dtype: DataType,
        shape: Vec<u64>,
    ) -> Result<Tensor<RunnerStorage>> {
        match &self.runner {
            Runner::V1(runner) => {
                for_each_carton_type! {
                    return match dtype {
                        $(
                            DataType::$CartonType =>
                                Ok(runner
                                    .alloc_tensor::<$RustType>(shape)
                                    .await
                                    .map_err(|e| CartonError::ErrorFromRunner(e))?
                                    .into()),
                        )*
                    }
                }
            }
        }
    }
}

#[cfg(not(target_family = "wasm"))]
#[cfg(test)]
mod tests {
    use std::time::Instant;

    use tokio::io::AsyncReadExt;

    #[tokio::test]
    async fn test_get() {
        let _ = env_logger::builder()
            .filter_level(log::LevelFilter::Info)
            .filter_module("carton", log::LevelFilter::Trace)
            .is_test(true)
            .try_init();

        let start = Instant::now();
        let info =
            super::Carton::get_model_info("https://carton.pub/cartonml/basic_example".to_owned())
                .await
                .unwrap();
        println!("Loaded model in {:#?}", start.elapsed());

        let start = Instant::now();
        let mut misc_file = info
            .info
            .misc_files
            .unwrap()
            .get("model_architecture.png")
            .unwrap()
            .get()
            .await;
        let mut buf = Vec::new();
        misc_file.read_to_end(&mut buf).await.unwrap();
        println!("Fetched misc file in {:#?}", start.elapsed());
    }

    /// This tests a subdomain of carton.pub to exercise a different code path
    /// We can remove this once the special case for carton.pub in `http.rs` is removed
    #[tokio::test]
    async fn test_other_domain() {
        let _ = env_logger::builder()
            .filter_level(log::LevelFilter::Info)
            .filter_module("carton", log::LevelFilter::Trace)
            .is_test(true)
            .try_init();

        let start = Instant::now();
        let _info =
            super::Carton::get_model_info("https://assets.carton.pub/manifest_sha256/0851b8cbda75c2f587c4c2a832c245575330a65932b9206f6e70391b78032c51")
                .await
                .unwrap();
        println!("Loaded info in {:#?}", start.elapsed());
    }
}
