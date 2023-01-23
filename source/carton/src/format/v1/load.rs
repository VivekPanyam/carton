//! Get a CartonInfo struct from a FS
//! This module does a lot of type conversions to map from the types in the toml file to the ones in
//! crate::types and crate::info

use std::collections::HashMap;
use std::sync::Arc;

use lunchbox::types::{MaybeSend, MaybeSync, ReadableFile};
use lunchbox::ReadableFileSystem;
use sha2::{Digest, Sha256};

use crate::conversion_utils::{convert_opt_map, convert_opt_vec, convert_vec};
use crate::error::{CartonError, Result};
use crate::info::{CartonInfoWithExtras, PossiblyLoaded};
use crate::types::CartonInfo;

async fn load_tensor_from_fs<T>(fs: &T, path: &str) -> crate::types::Tensor
where
    T: ReadableFileSystem,
    T::FileType: ReadableFile + 'static,
{
    let f = fs.open(path).await.unwrap();
    // Actually read the tensor and return it
    todo!()
}

async fn load_misc_from_fs<T>(fs: &T, path: &str) -> crate::info::MiscFile
where
    T: ReadableFileSystem,
    T::FileType: ReadableFile + MaybeSend + MaybeSync + 'static,
{
    Box::new(fs.open(path).await.unwrap())
}

pub(crate) async fn load<T>(fs: &Arc<T>) -> Result<CartonInfoWithExtras>
where
    T: ReadableFileSystem + MaybeSend + MaybeSync + 'static,
    T::FileType: ReadableFile + MaybeSend + MaybeSync + 'static,
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
                    let fs = fs.clone();
                    let owned_path = path.to_owned();
                    let val = PossiblyLoaded::from_loader(Box::pin(async move {
                        load_misc_from_fs(fs.as_ref(), &owned_path).await
                    }));

                    ("@".to_owned() + path, val)
                })
                .collect(),
        )
    };

    // Create a CartonInfo struct
    let info = CartonInfo {
        model_name: config.model_name,
        model_description: config.model_description,
        required_platforms: convert_opt_vec(config.required_platforms),
        inputs: convert_opt_vec(config.input),
        outputs: convert_opt_vec(config.output),
        self_tests: config.self_test.convert(fs),
        // TODO: reuse the misc files from above when loading examples
        examples: config.example.convert(fs),
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

// Type conversions
trait ConvertFrom<T> {
    fn from<F>(item: T, fs: &Arc<F>) -> Self
    where
        F: ReadableFileSystem + MaybeSend + MaybeSync + 'static,
        F::FileType: ReadableFile + MaybeSend + MaybeSync + 'static;
}

// Something like "into"
trait ConvertInto<T> {
    fn convert<F>(self, fs: &Arc<F>) -> T
    where
        F: ReadableFileSystem + MaybeSend + MaybeSync + 'static,
        F::FileType: ReadableFile + MaybeSend + MaybeSync + 'static;
}

// Blanket impl
impl<T, U> ConvertInto<U> for T
where
    U: ConvertFrom<T>,
{
    fn convert<F>(self, fs: &Arc<F>) -> U
    where
        F: ReadableFileSystem + MaybeSend + MaybeSync + 'static,
        F::FileType: ReadableFile + MaybeSend + MaybeSync + 'static,
    {
        U::from(self, fs)
    }
}

impl<T, U> ConvertFrom<Vec<T>> for Vec<U>
where
    U: ConvertFrom<T>,
{
    fn from<F>(item: Vec<T>, fs: &Arc<F>) -> Self
    where
        F: ReadableFileSystem + MaybeSend + MaybeSync + 'static,
        F::FileType: ReadableFile + MaybeSend + MaybeSync + 'static,
    {
        item.into_iter().map(|v| v.convert(fs)).collect()
    }
}

impl<T, U> ConvertFrom<HashMap<String, T>> for HashMap<String, U>
where
    U: ConvertFrom<T>,
{
    fn from<F>(item: HashMap<String, T>, fs: &Arc<F>) -> Self
    where
        F: ReadableFileSystem + MaybeSend + MaybeSync + 'static,
        F::FileType: ReadableFile + MaybeSend + MaybeSync + 'static,
    {
        item.into_iter().map(|(k, v)| (k, v.convert(fs))).collect()
    }
}

impl<T, U> ConvertFrom<Option<T>> for Option<U>
where
    U: ConvertFrom<T>,
{
    fn from<F>(item: Option<T>, fs: &Arc<F>) -> Self
    where
        F: ReadableFileSystem + MaybeSend + MaybeSync + 'static,
        F::FileType: ReadableFile + MaybeSend + MaybeSync + 'static,
    {
        item.map(|v| v.convert(fs))
    }
}

impl ConvertFrom<super::carton_toml::TensorReference> for PossiblyLoaded<crate::types::Tensor> {
    fn from<F>(item: super::carton_toml::TensorReference, fs: &Arc<F>) -> Self
    where
        F: ReadableFileSystem + MaybeSend + MaybeSync + 'static,
        F::FileType: ReadableFile + MaybeSend + MaybeSync + 'static,
    {
        let fs = fs.clone();
        PossiblyLoaded::from_loader(Box::pin(async move {
            load_tensor_from_fs(fs.as_ref(), item.0.strip_prefix("@").unwrap()).await
        }))
    }
}

impl ConvertFrom<super::carton_toml::MiscFileReference> for PossiblyLoaded<crate::info::MiscFile> {
    fn from<F>(item: super::carton_toml::MiscFileReference, fs: &Arc<F>) -> Self
    where
        F: ReadableFileSystem + MaybeSend + MaybeSync + 'static,
        F::FileType: ReadableFile + MaybeSend + MaybeSync + 'static,
    {
        let fs = fs.clone();
        PossiblyLoaded::from_loader(Box::pin(async move {
            load_misc_from_fs(fs.as_ref(), item.0.strip_prefix("@").unwrap()).await
        }))
    }
}

impl ConvertFrom<super::carton_toml::TensorOrMiscReference> for crate::info::TensorOrMisc {
    fn from<F>(item: super::carton_toml::TensorOrMiscReference, fs: &Arc<F>) -> Self
    where
        F: ReadableFileSystem + MaybeSend + MaybeSync + 'static,
        F::FileType: ReadableFile + MaybeSend + MaybeSync + 'static,
    {
        match item {
            super::carton_toml::TensorOrMiscReference::T(v) => {
                crate::info::TensorOrMisc::Tensor(v.convert(fs))
            }
            super::carton_toml::TensorOrMiscReference::M(v) => {
                crate::info::TensorOrMisc::Misc(v.convert(fs))
            }
        }
    }
}

impl ConvertFrom<super::carton_toml::Example> for crate::info::Example {
    fn from<F>(item: super::carton_toml::Example, fs: &Arc<F>) -> Self
    where
        F: ReadableFileSystem + MaybeSend + MaybeSync + 'static,
        F::FileType: ReadableFile + MaybeSend + MaybeSync + 'static,
    {
        Self {
            name: item.name,
            description: item.description,
            inputs: item.inputs.convert(fs),
            sample_out: item.sample_out.convert(fs),
        }
    }
}

impl ConvertFrom<super::carton_toml::SelfTest> for crate::info::SelfTest {
    fn from<F>(item: super::carton_toml::SelfTest, fs: &Arc<F>) -> Self
    where
        F: ReadableFileSystem + MaybeSend + MaybeSync + 'static,
        F::FileType: ReadableFile + MaybeSend + MaybeSync + 'static,
    {
        Self {
            name: item.name,
            description: item.description,
            inputs: item.inputs.convert(fs),
            expected_out: item.expected_out.convert(fs),
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
