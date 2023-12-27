//! This module contains types and conversion logic. Its a lot of code that's mostly tedious stuff
//! so hopefully there's a way to simplify this.
//!
//! It contains nodejs visible types that are similar to ones in `carton_core::info` along with type conversions
//! between the two.

use std::{collections::HashMap, str::FromStr, sync::Arc};

use async_trait::async_trait;
use carton_core::{
    conversion_utils::{convert_map, convert_opt_map, convert_opt_vec},
    info::{MiscFileLoader, PossiblyLoaded},
};
use napi::{
    bindgen_prelude::{Buffer, FromNapiValue, Null, ToNapiValue},
    tokio::io::AsyncReadExt,
    JsUnknown, NapiValue,
};

use crate::tensor::Tensor;

pub struct LoadOpts {
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
    pub visible_device: String,
}

pub enum RunnerOpt {
    Integer(i64),
    Double(f64),
    String(String),
    Boolean(bool),
}

impl ToNapiValue for RunnerOpt {
    unsafe fn to_napi_value(
        env: napi::sys::napi_env,
        val: Self,
    ) -> napi::Result<napi::sys::napi_value> {
        match val {
            RunnerOpt::Integer(val) => i64::to_napi_value(env, val),
            RunnerOpt::Double(val) => f64::to_napi_value(env, val),
            RunnerOpt::String(val) => String::to_napi_value(env, val),
            RunnerOpt::Boolean(val) => bool::to_napi_value(env, val),
        }
    }
}

impl FromNapiValue for RunnerOpt {
    fn from_unknown(value: JsUnknown) -> napi::Result<Self> {
        match value.get_type()? {
            napi::ValueType::Boolean => Ok(Self::Boolean(bool::from_unknown(value)?)),
            napi::ValueType::Number => {
                let val = f64::from_unknown(value)?;
                let trunc = val.trunc();
                // TODO: this can break if a float option happens to be an integer
                if trunc == val {
                    Ok(Self::Integer(trunc as _))
                } else {
                    Ok(Self::Double(val))
                }
            }
            napi::ValueType::String => Ok(Self::String(String::from_unknown(value)?)),
            other => panic!("Got invalid type in conversion: {other}"),
        }
    }

    unsafe fn from_napi_value(
        env: napi::sys::napi_env,
        napi_val: napi::sys::napi_value,
    ) -> napi::Result<Self> {
        Self::from_unknown(JsUnknown::from_napi_value(env, napi_val)?)
    }
}

impl From<RunnerOpt> for carton_core::types::RunnerOpt {
    fn from(value: RunnerOpt) -> Self {
        match value {
            RunnerOpt::Integer(v) => Self::Integer(v),
            RunnerOpt::Double(v) => Self::Double(v),
            RunnerOpt::String(v) => Self::String(v),
            RunnerOpt::Boolean(v) => Self::Boolean(v),
        }
    }
}

impl From<carton_core::info::RunnerOpt> for RunnerOpt {
    fn from(value: carton_core::info::RunnerOpt) -> Self {
        match value {
            carton_core::info::RunnerOpt::Integer(v) => Self::Integer(v),
            carton_core::info::RunnerOpt::Double(v) => Self::Double(v),
            carton_core::info::RunnerOpt::String(v) => Self::String(v),
            carton_core::info::RunnerOpt::Boolean(v) => Self::Boolean(v),
        }
    }
}

impl From<LoadOpts> for carton_core::types::LoadOpts {
    fn from(value: LoadOpts) -> Self {
        Self {
            override_runner_name: value.override_runner_name,
            override_required_framework_version: value.override_required_framework_version,
            override_runner_opts: convert_opt_map(value.override_runner_opts),
            visible_device: carton_core::types::Device::maybe_from_str(&value.visible_device)
                .unwrap(),
        }
    }
}

pub struct PackOpts {
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

    /// Information about the runner to use
    pub runner: RunnerInfo,

    /// Misc files that can be referenced by the description. The key is a normalized relative path
    /// (i.e one that does not reference parent directories, etc)
    pub misc_files: Option<HashMap<String, Buffer>>,
}

impl From<PackOpts> for carton_core::types::PackOpts {
    fn from(value: PackOpts) -> Self {
        let info = carton_core::types::CartonInfo {
            model_name: value.model_name,
            short_description: value.short_description,
            model_description: value.model_description,
            license: value.license,
            repository: value.repository,
            homepage: value.homepage,
            required_platforms: value.required_platforms.map(|v| {
                v.into_iter()
                    .map(|v| target_lexicon::Triple::from_str(&v).unwrap())
                    .collect()
            }),
            inputs: convert_opt_vec(value.inputs),
            outputs: convert_opt_vec(value.outputs),
            self_tests: convert_opt_vec(value.self_tests),
            examples: convert_opt_vec(value.examples),
            runner: value.runner.into(),
            misc_files: value.misc_files.map(|v| {
                v.into_iter()
                    .map(|(k, v)| (k, Arc::new(NodeMiscFileLoader(Arc::new(v))) as _))
                    .collect()
            }),
        };

        Self {
            info,
            // TODO: allow specifying linked files
            linked_files: None,
        }
    }
}

#[napi(object)]
pub struct TensorSpec {
    pub name: String,

    /// The datatype
    pub dtype: String,

    /// Tensor shape
    // Note: this is marked as optional here, but that's an implementation detail
    // It should be explicitly provided as `null`
    pub shape: Option<NodeShape>,

    /// Optional description
    pub description: Option<String>,

    /// Optional internal name
    pub internal_name: Option<String>,
}

pub struct NodeShape {
    inner: carton_core::info::Shape,
}

impl ToNapiValue for NodeShape {
    unsafe fn to_napi_value(
        env: napi::sys::napi_env,
        val: Self,
    ) -> napi::Result<napi::sys::napi_value> {
        Ok(match val.inner {
            carton_core::info::Shape::Any => Null::to_napi_value(env, Null)?,
            carton_core::info::Shape::Symbol(val) => String::to_napi_value(env, val)?,
            carton_core::info::Shape::Shape(val) => {
                let mut out = Vec::new();
                for item in val {
                    out.push(convert_from_dimension(env, item)?);
                }

                Vec::to_napi_value(env, out)?
            }
        })
    }
}

impl FromNapiValue for NodeShape {
    fn from_unknown(value: JsUnknown) -> napi::Result<Self> {
        if value.is_array().unwrap() {
            // A vec of dimensions
            let obj = value.coerce_to_object().unwrap();
            let arr_len = obj.get_array_length_unchecked().unwrap();
            let mut out = Vec::new();
            for i in 0..arr_len {
                let item: JsUnknown = obj.get_element(i).unwrap();
                out.push(convert_dimension(item));
            }

            Ok(Self {
                inner: carton_core::info::Shape::Shape(out),
            })
        } else {
            match value.get_type().unwrap() {
                napi::ValueType::Undefined | napi::ValueType::Null => Ok(Self {
                    inner: carton_core::info::Shape::Any,
                }),
                napi::ValueType::String => Ok(Self {
                    inner: carton_core::info::Shape::Symbol(String::from_unknown(value).unwrap()),
                }),
                other => panic!("Got unexpected type in conversions: {other}"),
            }
        }
    }

    unsafe fn from_napi_value(
        env: napi::sys::napi_env,
        napi_val: napi::sys::napi_value,
    ) -> napi::Result<Self> {
        Self::from_unknown(JsUnknown::from_napi_value(env, napi_val)?)
    }
}

impl From<TensorSpec> for carton_core::info::TensorSpec {
    fn from(value: TensorSpec) -> Self {
        Self {
            name: value.name,
            dtype: carton_core::info::DataType::from_str(&value.dtype).unwrap(),
            shape: value
                .shape
                .map(|v| v.inner)
                .unwrap_or(carton_core::info::Shape::Any),
            description: value.description,
            internal_name: value.internal_name,
        }
    }
}

impl From<carton_core::info::TensorSpec> for TensorSpec {
    fn from(value: carton_core::info::TensorSpec) -> Self {
        Self {
            name: value.name,
            dtype: value.dtype.to_str().to_owned(),
            shape: Some(NodeShape { inner: value.shape }),
            description: value.description,
            internal_name: value.internal_name,
        }
    }
}

fn convert_dimension(value: JsUnknown) -> carton_core::info::Dimension {
    match value.get_type().unwrap() {
        napi::ValueType::Undefined | napi::ValueType::Null => carton_core::info::Dimension::Any,
        napi::ValueType::Number => {
            carton_core::info::Dimension::Value(u32::from_unknown(value).unwrap() as _)
        }
        napi::ValueType::String => {
            carton_core::info::Dimension::Symbol(String::from_unknown(value).unwrap())
        }
        other => panic!("Got unexpected type in conversions: {other}"),
    }
}

unsafe fn convert_from_dimension(
    env: napi::sys::napi_env,
    value: carton_core::info::Dimension,
) -> napi::Result<JsUnknown> {
    match value {
        carton_core::info::Dimension::Value(val) => {
            let val = u32::to_napi_value(env, val as _)?;
            JsUnknown::from_raw(env, val)
        }
        carton_core::info::Dimension::Symbol(val) => {
            let val = String::to_napi_value(env, val)?;
            JsUnknown::from_raw(env, val)
        }
        carton_core::info::Dimension::Any => {
            let val = Null::to_napi_value(env, Null)?;
            JsUnknown::from_raw(env, val)
        }
    }
}

#[napi(object)]
pub struct RunnerInfo {
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
    pub opts: Option<HashMap<String, RunnerOpt>>,
}

impl From<RunnerInfo> for carton_core::info::RunnerInfo {
    fn from(value: RunnerInfo) -> Self {
        Self {
            runner_name: value.runner_name,
            required_framework_version: semver::VersionReq::from_str(
                &value.required_framework_version,
            )
            .unwrap(),
            runner_compat_version: value.runner_compat_version.map(|v| v as _),
            opts: convert_opt_map(value.opts),
        }
    }
}

impl From<carton_core::info::RunnerInfo> for RunnerInfo {
    fn from(value: carton_core::info::RunnerInfo) -> Self {
        Self {
            runner_name: value.runner_name,
            required_framework_version: value.required_framework_version.to_string(),
            runner_compat_version: value.runner_compat_version.map(|v| v as _),
            opts: convert_opt_map(value.opts),
        }
    }
}

#[napi(object)]
pub struct PackSelfTest {
    pub name: Option<String>,
    pub description: Option<String>,
    pub inputs: HashMap<String, Tensor>,

    // Can be empty
    pub expected_out: Option<HashMap<String, Tensor>>,
}

impl From<PackSelfTest> for carton_core::info::SelfTest {
    fn from(value: PackSelfTest) -> Self {
        Self {
            name: value.name,
            description: value.description,
            inputs: convert_map(value.inputs),
            expected_out: convert_opt_map(value.expected_out),
        }
    }
}

impl From<Tensor> for PossiblyLoaded<carton_core::types::Tensor> {
    fn from(value: Tensor) -> Self {
        Self::from_value(value.into())
    }
}

#[napi(object)]
pub struct PackExample {
    pub name: Option<String>,
    pub description: Option<String>,
    pub inputs: HashMap<String, PackTensorOrMisc>,
    pub sample_out: HashMap<String, PackTensorOrMisc>,
}

impl From<PackExample> for carton_core::info::Example {
    fn from(value: PackExample) -> Self {
        Self {
            name: value.name,
            description: value.description,
            inputs: convert_map(value.inputs),
            sample_out: convert_map(value.sample_out),
        }
    }
}

#[derive(Clone)]
pub enum PackTensorOrMisc {
    T(Tensor),
    M(Buffer),
}

struct NodeMiscFileLoader(Arc<Buffer>);
struct NodeMiscFile(Arc<Buffer>);

impl AsRef<[u8]> for NodeMiscFile {
    fn as_ref(&self) -> &[u8] {
        self.0.as_ref()
    }
}

#[async_trait]
impl MiscFileLoader for NodeMiscFileLoader {
    async fn get(&self) -> carton_core::info::MiscFile {
        Box::new(std::io::Cursor::new(NodeMiscFile(self.0.clone())))
    }
}

// TODO: describe why this is safe
unsafe impl Send for NodeMiscFile {}
unsafe impl Sync for NodeMiscFile {}
unsafe impl Send for NodeMiscFileLoader {}
unsafe impl Sync for NodeMiscFileLoader {}

impl From<PackTensorOrMisc> for carton_core::info::TensorOrMisc {
    fn from(value: PackTensorOrMisc) -> Self {
        match value {
            PackTensorOrMisc::T(t) => Self::Tensor(t.into()),
            PackTensorOrMisc::M(v) => Self::Misc(Arc::new(NodeMiscFileLoader(Arc::new(v)))),
        }
    }
}

impl ToNapiValue for PackTensorOrMisc {
    unsafe fn to_napi_value(
        env: napi::sys::napi_env,
        val: Self,
    ) -> napi::Result<napi::sys::napi_value> {
        match val {
            PackTensorOrMisc::T(val) => Tensor::to_napi_value(env, val),
            PackTensorOrMisc::M(val) => Buffer::to_napi_value(env, val),
        }
    }
}

impl FromNapiValue for PackTensorOrMisc {
    fn from_unknown(value: JsUnknown) -> napi::Result<Self> {
        if value.is_buffer()? {
            Ok(Self::M(Buffer::from_unknown(value)?))
        } else {
            Ok(Self::T(Tensor::from_unknown(value)?))
        }
    }

    unsafe fn from_napi_value(
        env: napi::sys::napi_env,
        napi_val: napi::sys::napi_value,
    ) -> napi::Result<Self> {
        Self::from_unknown(JsUnknown::from_napi_value(env, napi_val)?)
    }
}

// -----

#[napi(object)]
pub struct CartonInfo {
    /// The name of the model
    pub model_name: Option<String>,

    /// A short description (should be 100 characters or less)
    pub short_description: Option<String>,

    /// The model description
    pub model_description: Option<String>,

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
    pub self_tests: Option<Vec<InfoSelfTest>>,

    /// Examples
    /// Can be empty
    pub examples: Option<Vec<InfoExample>>,

    /// Information about the runner to use
    pub runner: RunnerInfo,

    /// Misc files that can be referenced by the description. The key is a normalized relative path
    /// (i.e one that does not reference parent directories, etc)
    pub misc_files: Option<HashMap<String, ArcMiscFileLoaderWrapper>>,
}

impl From<carton_core::info::CartonInfo> for CartonInfo {
    fn from(value: carton_core::info::CartonInfo) -> Self {
        Self {
            model_name: value.model_name,
            short_description: value.short_description,
            model_description: value.model_description,
            required_platforms: value
                .required_platforms
                .map(|v| v.into_iter().map(|v| v.to_string()).collect()),
            inputs: convert_opt_vec(value.inputs),
            outputs: convert_opt_vec(value.outputs),
            self_tests: convert_opt_vec(value.self_tests),
            examples: convert_opt_vec(value.examples),
            runner: value.runner.into(),
            misc_files: convert_opt_map(value.misc_files),
        }
    }
}

#[napi]
pub struct ArcMiscFileLoaderWrapper {
    inner: carton_core::info::ArcMiscFileLoader,
}

impl FromNapiValue for ArcMiscFileLoaderWrapper {
    unsafe fn from_napi_value(
        _env: napi::sys::napi_env,
        _napi_val: napi::sys::napi_value,
    ) -> napi::Result<Self> {
        panic!("Tried calling fromnapivalue for ArcMiscFileLoaderWrapper")
    }
}

#[napi]
impl ArcMiscFileLoaderWrapper {
    #[napi]
    pub async fn read(&self) -> Buffer {
        let mut f = self.inner.get().await;

        // Not ideal, but we'll just read the whole file and turn it into a buffer
        let mut v = Vec::new();
        f.read_to_end(&mut v).await.unwrap();
        v.into()
    }
}

impl From<carton_core::info::ArcMiscFileLoader> for ArcMiscFileLoaderWrapper {
    fn from(value: carton_core::info::ArcMiscFileLoader) -> Self {
        Self { inner: value }
    }
}

#[napi]
#[derive(Clone)]
pub struct PossiblyLoadedWrapper {
    inner: PossiblyLoaded<Tensor>,
}

impl FromNapiValue for PossiblyLoadedWrapper {
    unsafe fn from_napi_value(
        _env: napi::sys::napi_env,
        _napi_val: napi::sys::napi_value,
    ) -> napi::Result<Self> {
        panic!("Tried calling fromnapivalue for PossiblyLoadedWrapper")
    }
}

#[napi]
impl PossiblyLoadedWrapper {
    #[napi]
    pub async fn get(&self) -> Tensor {
        // TODO: this probably isn't ideal
        self.inner.get().await.clone()
    }
}

impl From<PossiblyLoaded<carton_core::types::Tensor>> for PossiblyLoadedWrapper {
    fn from(value: PossiblyLoaded<carton_core::types::Tensor>) -> Self {
        Self {
            inner: PossiblyLoaded::from_loader(Box::pin(async move {
                let t = value.get().await;

                t.into()
            })),
        }
    }
}

#[napi(object)]
pub struct InfoSelfTest {
    pub name: Option<String>,
    pub description: Option<String>,
    pub inputs: HashMap<String, PossiblyLoadedWrapper>,

    // Can be empty
    pub expected_out: Option<HashMap<String, PossiblyLoadedWrapper>>,
}

impl From<carton_core::info::SelfTest> for InfoSelfTest {
    fn from(value: carton_core::info::SelfTest) -> Self {
        Self {
            name: value.name,
            description: value.description,
            inputs: convert_map(value.inputs),
            expected_out: convert_opt_map(value.expected_out),
        }
    }
}

#[napi(object)]
pub struct InfoExample {
    pub name: Option<String>,
    pub description: Option<String>,
    pub inputs: HashMap<String, InfoTensorOrMisc>,
    pub sample_out: HashMap<String, InfoTensorOrMisc>,
}

impl From<carton_core::info::Example> for InfoExample {
    fn from(value: carton_core::info::Example) -> Self {
        Self {
            name: value.name,
            description: value.description,
            inputs: convert_map(value.inputs),
            sample_out: convert_map(value.sample_out),
        }
    }
}

pub enum InfoTensorOrMisc {
    T(PossiblyLoadedWrapper),
    M(ArcMiscFileLoaderWrapper),
}

impl FromNapiValue for InfoTensorOrMisc {
    unsafe fn from_napi_value(
        _env: napi::sys::napi_env,
        _napi_val: napi::sys::napi_value,
    ) -> napi::Result<Self> {
        panic!("Tried calling fromnapivalue for InfoTensorOrMisc")
    }
}

impl ToNapiValue for InfoTensorOrMisc {
    unsafe fn to_napi_value(
        env: napi::sys::napi_env,
        val: Self,
    ) -> napi::Result<napi::sys::napi_value> {
        match val {
            InfoTensorOrMisc::T(val) => PossiblyLoadedWrapper::to_napi_value(env, val),
            InfoTensorOrMisc::M(val) => ArcMiscFileLoaderWrapper::to_napi_value(env, val),
        }
    }
}

impl From<carton_core::info::TensorOrMisc> for InfoTensorOrMisc {
    fn from(value: carton_core::info::TensorOrMisc) -> Self {
        match value {
            carton_core::info::TensorOrMisc::Tensor(t) => Self::T(t.into()),
            carton_core::info::TensorOrMisc::Misc(m) => Self::M(m.into()),
        }
    }
}
