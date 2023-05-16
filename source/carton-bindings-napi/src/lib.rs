use std::collections::HashMap;

use napi::bindgen_prelude::{Buffer, External, Result};
use tensor::Tensor;
use types::{CartonInfo, LoadOpts, PackOpts, RunnerInfo};

use crate::types::{PackExample, PackSelfTest, RunnerOpt, TensorSpec};

#[macro_use]
extern crate napi_derive;

mod tensor;
mod types;

#[napi(object)]
pub struct LoadArgs {
    /// A URL or path to the carton to load
    pub url_or_path: String,

    /// Override the runner to use
    /// If not overridden, this is fetched from the carton metadata
    pub override_runner_name: Option<String>,

    /// Override the framework_version to use
    /// If not overridden, this is fetched from the carton metadata
    pub override_required_framework_version: Option<String>,

    /// Options to pass to the runner. These are runner-specific (e.g.
    /// PyTorch, TensorFlow, etc).
    ///
    /// Overrides are merged with the options set in the carton metadata
    /// Sometimes used to configure thread-pool sizes, etc.
    /// See the documentation for more info
    pub override_runner_opts: Option<HashMap<String, RunnerOpt>>,

    /// The device that is visible to this model.
    /// Note: a visible device does not necessarily mean that the model
    /// will use that device; it is up to the model to actually use it
    /// (e.g. by moving itself to GPU if it sees one available)
    pub visible_device: Option<String>,
}

#[napi]
pub async fn load(args: LoadArgs) -> Result<Carton> {
    let opts = LoadOpts {
        override_runner_name: args.override_runner_name,
        override_required_framework_version: args.override_required_framework_version,
        override_runner_opts: args.override_runner_opts,
        visible_device: args.visible_device.unwrap_or("cpu".to_owned()),
    };

    let inner = carton_core::Carton::load(args.url_or_path, opts.into())
        .await
        .unwrap();

    Ok(Carton { inner })
}

#[napi(object)]
pub struct PackArgs {
    /// A path to the model to pack
    pub path: String,

    /// The name of the runner to use
    pub runner_name: String,

    /// The required framework version range to run the model with
    /// This is a semver version range. See https://docs.rs/semver/1.0.16/semver/struct.VersionReq.html
    /// for format details.
    /// For example `=1.2.4`, means exactly version `1.2.4`
    /// In most cases, this should be exactly one version
    pub required_framework_version: String,

    /// Don't set this unless you know what you're doing
    pub runner_compat_version: Option<u32>,

    /// Options to pass to the runner. These are runner-specific (e.g.
    /// PyTorch, TensorFlow, etc).
    ///
    /// Sometimes used to configure thread-pool sizes, etc.
    /// See the documentation for more info
    pub runner_opts: Option<HashMap<String, RunnerOpt>>,

    /// The name of the model
    pub model_name: Option<String>,

    /// A short description (should be 100 characters or less)
    pub short_description: Option<String>,

    /// The model description
    pub model_description: Option<String>,

    /// The license for this model. This should be an SPDX expression, but may not be
    /// for non-SPDX license types.
    pub license: Option<String>,

    /// A URL for a repository for this model
    pub repository: Option<String>,

    /// A URL for a website that is the homepage for this model
    pub homepage: Option<String>,

    /// A list of platforms this model supports
    /// If empty or unspecified, all platforms are okay
    pub required_platforms: Option<Vec<String>>,

    /// A list of inputs for the model
    /// Can be empty
    pub inputs: Option<Vec<TensorSpec>>,

    /// A list of outputs for the model
    /// Can be empty
    pub outputs: Option<Vec<TensorSpec>>,

    /// Test data
    /// Can be empty
    pub self_tests: Option<Vec<PackSelfTest>>,

    /// Examples
    /// Can be empty
    pub examples: Option<Vec<PackExample>>,

    /// Misc files that can be referenced by the description. The key is a normalized relative path
    /// (i.e one that does not reference parent directories, etc)
    pub misc_files: Option<HashMap<String, Buffer>>,
}

#[napi]
pub async fn pack(args: PackArgs) -> Result<String> {
    let opts = PackOpts {
        model_name: args.model_name,
        short_description: args.short_description,
        model_description: args.model_description,
        license: args.license,
        repository: args.repository,
        homepage: args.homepage,
        required_platforms: args.required_platforms,
        inputs: args.inputs,
        outputs: args.outputs,
        self_tests: args.self_tests,
        examples: args.examples,
        runner: RunnerInfo {
            runner_name: args.runner_name,
            required_framework_version: args.required_framework_version,
            runner_compat_version: args.runner_compat_version,
            opts: args.runner_opts,
        },
        misc_files: args.misc_files,
    };

    let out = carton_core::Carton::pack(args.path, opts).await.unwrap();

    Ok(out.to_str().unwrap().to_owned())
}

#[napi(object)]
pub struct LoadUnpackedArgs {
    /// A path to the model to load
    pub path: String,

    /// The name of the runner to use
    pub runner_name: String,

    /// The required framework version range to run the model with
    /// This is a semver version range. See https://docs.rs/semver/1.0.16/semver/struct.VersionReq.html
    /// for format details.
    /// For example `=1.2.4`, means exactly version `1.2.4`
    /// In most cases, this should be exactly one version
    pub required_framework_version: String,

    /// Don't set this unless you know what you're doing
    pub runner_compat_version: Option<u32>,

    /// Options to pass to the runner. These are runner-specific (e.g.
    /// PyTorch, TensorFlow, etc).
    ///
    /// Sometimes used to configure thread-pool sizes, etc.
    /// See the documentation for more info
    pub runner_opts: Option<HashMap<String, RunnerOpt>>,

    /// The name of the model
    pub model_name: Option<String>,

    /// A short description (should be 100 characters or less)
    pub short_description: Option<String>,

    /// The model description
    pub model_description: Option<String>,

    /// The license for this model. This should be an SPDX expression, but may not be
    /// for non-SPDX license types.
    pub license: Option<String>,

    /// A URL for a repository for this model
    pub repository: Option<String>,

    /// A URL for a website that is the homepage for this model
    pub homepage: Option<String>,

    /// A list of platforms this model supports
    /// If empty or unspecified, all platforms are okay
    pub required_platforms: Option<Vec<String>>,

    /// A list of inputs for the model
    /// Can be empty
    pub inputs: Option<Vec<TensorSpec>>,

    /// A list of outputs for the model
    /// Can be empty
    pub outputs: Option<Vec<TensorSpec>>,

    /// Test data
    /// Can be empty
    pub self_tests: Option<Vec<PackSelfTest>>,

    /// Examples
    /// Can be empty
    pub examples: Option<Vec<PackExample>>,

    /// Misc files that can be referenced by the description. The key is a normalized relative path
    /// (i.e one that does not reference parent directories, etc)
    pub misc_files: Option<HashMap<String, Buffer>>,

    /// The device that is visible to this model.
    /// Note: a visible device does not necessarily mean that the model
    /// will use that device; it is up to the model to actually use it
    /// (e.g. by moving itself to GPU if it sees one available)
    pub visible_device: Option<String>,
}

#[napi]
pub async fn load_unpacked(args: LoadUnpackedArgs) -> Result<Carton> {
    let pack_opts = PackOpts {
        model_name: args.model_name,
        short_description: args.short_description,
        model_description: args.model_description,
        license: args.license,
        repository: args.repository,
        homepage: args.homepage,
        required_platforms: args.required_platforms,
        inputs: args.inputs,
        outputs: args.outputs,
        self_tests: args.self_tests,
        examples: args.examples,
        runner: RunnerInfo {
            runner_name: args.runner_name,
            required_framework_version: args.required_framework_version,
            runner_compat_version: args.runner_compat_version,
            opts: args.runner_opts,
        },
        misc_files: args.misc_files,
    };

    let load_opts = LoadOpts {
        override_runner_name: None,
        override_required_framework_version: None,
        override_runner_opts: None,
        visible_device: args.visible_device.unwrap_or("cpu".to_owned()),
    };

    let inner = carton_core::Carton::load_unpacked(args.path, pack_opts, load_opts.into())
        .await
        .unwrap();

    Ok(Carton { inner })
}

type SealHandle = External<carton_core::types::SealHandle>;

#[napi]
pub struct Carton {
    inner: carton_core::Carton,
}

#[napi]
impl Carton {
    #[napi]
    pub async fn infer(&self, tensors: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
        let tensors = carton_core::conversion_utils::convert_map(tensors);
        let res = self.inner.infer(tensors).await.unwrap();
        Ok(carton_core::conversion_utils::convert_map(res))
    }

    #[napi]
    pub async fn seal(&self, tensors: HashMap<String, Tensor>) -> Result<SealHandle> {
        let tensors = carton_core::conversion_utils::convert_map(tensors);
        let res = self.inner.seal(tensors).await.unwrap();
        Ok(res.into())
    }

    #[napi]
    pub async fn infer_with_handle(&self, handle: SealHandle) -> Result<HashMap<String, Tensor>> {
        let handle = *handle;
        let res = self.inner.infer_with_handle(handle).await.unwrap();
        Ok(carton_core::conversion_utils::convert_map(res))
    }

    #[napi]
    pub fn get_info(&mut self) -> CartonInfo {
        self.inner.get_info().info.clone().into()
    }

    #[napi]
    pub async fn get_model_info(url_or_path: String) -> Result<CartonInfo> {
        let info = carton_core::Carton::get_model_info(url_or_path)
            .await
            .unwrap()
            .info;
        Ok(info.into())
    }
}
