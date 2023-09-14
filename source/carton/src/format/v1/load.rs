//! Get a CartonInfo struct from a FS
//! This module does a lot of type conversions to map from the types in the toml file to the ones in
//! crate::types and crate::info

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use lunchbox::types::{MaybeSend, MaybeSync, ReadableFile};
use lunchbox::ReadableFileSystem;
use sha2::{Digest, Sha256};

use crate::conversion_utils::{
    convert_opt_map, convert_opt_vec, convert_vec, ConvertFromWithContext, ConvertIntoWithContext,
};
use crate::error::{CartonError, Result};
use crate::info::{CartonInfoWithExtras, PossiblyLoaded};
use crate::types::{CartonInfo, GenericStorage};

struct MiscFileLoader<T> {
    fs: Arc<T>,
    path: String,
}

#[cfg_attr(target_family = "wasm", async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait)]
impl<T> crate::info::MiscFileLoader for MiscFileLoader<T>
where
    T: ReadableFileSystem + MaybeSend + MaybeSync,
    T::FileType: ReadableFile + MaybeSend + MaybeSync + Unpin + 'static,
{
    async fn get(&self) -> crate::info::MiscFile {
        Box::new(self.fs.open(&self.path).await.unwrap())
    }
}

pub(crate) async fn load<T>(fs: &Arc<T>) -> Result<CartonInfoWithExtras<GenericStorage>>
where
    T: ReadableFileSystem + MaybeSend + MaybeSync + 'static,
    T::FileType: ReadableFile + MaybeSend + MaybeSync + Unpin + 'static,
{
    // Load the toml file
    let toml = fs.read("/carton.toml").await?;
    let config = crate::format::v1::carton_toml::parse(&toml).await?;

    // Check for misc files
    let manifest = fs.read_to_string("/MANIFEST").await?;
    let mut misc_file_paths = Vec::new();

    // Filter the manifest to files in `misc/`
    // Note: not using `filter` so we can return errors easily
    for line in manifest.lines() {
        if let Some((file_path, _sha256)) = line.rsplit_once("=") {
            if file_path.starts_with("misc/") {
                misc_file_paths.push(file_path);
            }
        } else {
            return Err(CartonError::Other(
                "MANIFEST was not in the form {path}={sha256}",
            ));
        }
    }

    // Create the loaders for all the misc files
    let misc_files = if misc_file_paths.is_empty() {
        None
    } else {
        Some(
            misc_file_paths
                .into_iter()
                .map(|path| {
                    let mfl = MiscFileLoader {
                        fs: fs.clone(),
                        path: path.to_owned(),
                    };

                    let mfl: crate::info::ArcMiscFileLoader = Arc::new(mfl);

                    (path.strip_prefix("misc/").unwrap().to_owned(), mfl)
                })
                .collect(),
        )
    };

    let tensors =
        super::tensor::load_tensors(fs, lunchbox::path::Path::new("tensor_data/")).await?;
    let load_context = LoadContext { fs, tensors };

    // Create a CartonInfo struct
    let info = CartonInfo {
        model_name: config.model_name,
        short_description: config.short_description,
        model_description: config.model_description,
        license: config.license,
        repository: config.repository,
        homepage: config.homepage,
        required_platforms: convert_opt_vec(config.required_platforms),
        inputs: convert_opt_vec(config.input),
        outputs: convert_opt_vec(config.output),
        self_tests: config.self_test.convert_into_with_context(&load_context),
        // TODO: reuse the misc files from above when loading examples
        examples: config.example.convert_into_with_context(&load_context),
        runner: config.runner.into(),
        misc_files,
    };

    // Compute the manifest sha256
    let manifest = fs.read("/MANIFEST").await?;
    let mut hasher = Sha256::new();
    hasher.update(manifest);
    let manifest_sha256 = Some(format!("{:x}", hasher.finalize()));

    Ok(CartonInfoWithExtras {
        info,
        manifest_sha256,
    })
}

struct LoadContext<'a, F> {
    fs: &'a Arc<F>,
    tensors: HashMap<String, PossiblyLoaded<crate::types::Tensor<GenericStorage>>>,
}

impl<'a, F> ConvertFromWithContext<super::carton_toml::TensorReference, &LoadContext<'a, F>>
    for PossiblyLoaded<crate::types::Tensor<GenericStorage>>
where
    F: ReadableFileSystem + MaybeSend + MaybeSync + 'static,
    F::FileType: ReadableFile + MaybeSend + MaybeSync + 'static,
{
    fn from(item: super::carton_toml::TensorReference, context: &LoadContext<F>) -> Self {
        let key = item.0.strip_prefix("@tensor_data/").unwrap();
        context.tensors[key].clone()
    }
}

impl<'a, F> ConvertFromWithContext<super::carton_toml::MiscFileReference, &LoadContext<'a, F>>
    for crate::info::ArcMiscFileLoader
where
    F: ReadableFileSystem + MaybeSend + MaybeSync + 'static,
    F::FileType: ReadableFile + MaybeSend + MaybeSync + Unpin + 'static,
{
    fn from(item: super::carton_toml::MiscFileReference, context: &LoadContext<F>) -> Self {
        let mfl = MiscFileLoader {
            fs: context.fs.clone(),
            path: item.0.strip_prefix("@").unwrap().to_owned(),
        };

        Arc::new(mfl)
    }
}

impl<C> ConvertFromWithContext<super::carton_toml::TensorOrMiscReference, C>
    for crate::info::TensorOrMisc<GenericStorage>
where
    C: Copy,
    PossiblyLoaded<crate::types::Tensor<GenericStorage>>:
        ConvertFromWithContext<super::carton_toml::TensorReference, C>,
    crate::info::ArcMiscFileLoader:
        ConvertFromWithContext<super::carton_toml::MiscFileReference, C>,
{
    fn from(item: super::carton_toml::TensorOrMiscReference, context: C) -> Self {
        match item {
            super::carton_toml::TensorOrMiscReference::T(v) => {
                crate::info::TensorOrMisc::Tensor(v.convert_into_with_context(context))
            }
            super::carton_toml::TensorOrMiscReference::M(v) => {
                crate::info::TensorOrMisc::Misc(v.convert_into_with_context(context))
            }
        }
    }
}

impl<C> ConvertFromWithContext<super::carton_toml::Example, C>
    for crate::info::Example<GenericStorage>
where
    C: Copy,
    crate::info::TensorOrMisc<GenericStorage>:
        ConvertFromWithContext<super::carton_toml::TensorOrMiscReference, C>,
{
    fn from(item: super::carton_toml::Example, context: C) -> Self {
        Self {
            name: item.name,
            description: item.description,
            inputs: item.inputs.convert_into_with_context(context),
            sample_out: item.sample_out.convert_into_with_context(context),
        }
    }
}

impl<C> ConvertFromWithContext<super::carton_toml::SelfTest, C>
    for crate::info::SelfTest<GenericStorage>
where
    C: Copy,
    PossiblyLoaded<crate::types::Tensor<GenericStorage>>:
        ConvertFromWithContext<super::carton_toml::TensorReference, C>,
{
    fn from(item: super::carton_toml::SelfTest, context: C) -> Self {
        Self {
            name: item.name,
            description: item.description,
            inputs: item.inputs.convert_into_with_context(context),
            expected_out: item.expected_out.convert_into_with_context(context),
        }
    }
}

impl From<super::carton_toml::Triple> for target_lexicon::Triple {
    fn from(value: super::carton_toml::Triple) -> Self {
        value.0
    }
}

impl From<super::carton_toml::TensorSpec> for crate::info::TensorSpec {
    fn from(value: super::carton_toml::TensorSpec) -> Self {
        Self {
            name: value.name,
            dtype: value.dtype.into(),
            shape: value.shape.into(),
            description: value.description,
            internal_name: value.internal_name,
        }
    }
}

impl From<super::carton_toml::RunnerInfo> for crate::info::RunnerInfo {
    fn from(value: super::carton_toml::RunnerInfo) -> Self {
        Self {
            runner_name: value.runner_name,
            required_framework_version: value.required_framework_version,
            runner_compat_version: Some(value.runner_compat_version),
            opts: convert_opt_map(value.opts),
        }
    }
}

impl From<super::carton_toml::Shape> for crate::info::Shape {
    fn from(value: super::carton_toml::Shape) -> Self {
        match value {
            super::carton_toml::Shape::Any => Self::Any,
            super::carton_toml::Shape::Symbol(v) => Self::Symbol(v),
            super::carton_toml::Shape::Shape(v) => Self::Shape(convert_vec(v)),
        }
    }
}

impl From<super::carton_toml::Dimension> for crate::info::Dimension {
    fn from(value: super::carton_toml::Dimension) -> Self {
        match value {
            super::carton_toml::Dimension::Value(v) => Self::Value(v),
            super::carton_toml::Dimension::Symbol(v) => Self::Symbol(v),
            super::carton_toml::Dimension::Any => Self::Any,
        }
    }
}

impl From<super::carton_toml::DataType> for crate::info::DataType {
    fn from(value: super::carton_toml::DataType) -> Self {
        match value {
            super::carton_toml::DataType::Float32 => Self::Float,
            super::carton_toml::DataType::Float64 => Self::Double,
            super::carton_toml::DataType::String => Self::String,
            super::carton_toml::DataType::Int8 => Self::I8,
            super::carton_toml::DataType::Int16 => Self::I16,
            super::carton_toml::DataType::Int32 => Self::I32,
            super::carton_toml::DataType::Int64 => Self::I64,
            super::carton_toml::DataType::Uint8 => Self::U8,
            super::carton_toml::DataType::Uint16 => Self::U16,
            super::carton_toml::DataType::Uint32 => Self::U32,
            super::carton_toml::DataType::Uint64 => Self::U64,
        }
    }
}

impl From<super::carton_toml::RunnerOpt> for crate::info::RunnerOpt {
    fn from(value: super::carton_toml::RunnerOpt) -> Self {
        match value {
            super::carton_toml::RunnerOpt::Integer(v) => Self::Integer(v),
            super::carton_toml::RunnerOpt::Double(v) => Self::Double(v),
            super::carton_toml::RunnerOpt::String(v) => Self::String(v),
            super::carton_toml::RunnerOpt::Boolean(v) => Self::Boolean(v),
        }
    }
}
