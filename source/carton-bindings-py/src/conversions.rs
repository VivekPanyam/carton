//! This module implements conversions between the types in the core carton library and the ones in the python bindings
//! Most of this is boilerplate so maybe there's a clean way to remove some portion of the code below
use std::sync::Arc;
use std::{collections::HashMap, str::FromStr};

use async_trait::async_trait;
use carton_core::conversion_utils::{convert_map, convert_opt_map, convert_opt_vec, convert_vec};
use carton_core::info::LinkedFile;
use carton_core::types::{DataType, GenericStorage, RunnerOpt, Tensor};
use numpy::ToPyArray;
use pyo3::types::PyBytes;
use pyo3::{exceptions::PyValueError, prelude::*, PyDowncastError};
use semver::VersionReq;
use target_lexicon::Triple;
use tokio::io::AsyncReadExt;
use tokio::sync::Mutex;

use crate::tensor::{tensor_to_py, PyTensorStorage, SupportedTensorType};

pub(crate) fn create_load_opts(
    visible_device: Option<Device>,
    override_runner_name: Option<String>,
    override_required_framework_version: Option<String>,
    override_runner_opts: Option<HashMap<String, PyRunnerOpt>>,
) -> PyResult<carton_core::types::LoadOpts> {
    Ok(carton_core::types::LoadOpts {
        override_runner_name,
        override_required_framework_version,
        override_runner_opts: convert_opt_map(override_runner_opts),
        // TODO: use something more specific than ValueError
        visible_device: match visible_device {
            None => carton_core::types::Device::default(),
            Some(v) => match v {
                Device::Int(v) => carton_core::types::Device::maybe_from_index(v),
                Device::String(v) => carton_core::types::Device::maybe_from_str(&v)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?,
            },
        },
    })
}

#[derive(FromPyObject)]
pub(crate) enum Device {
    Int(u32),
    String(String),
}

pub(crate) fn create_pack_opts(
    runner_name: String,
    required_framework_version: String,
    runner_compat_version: Option<u64>,
    runner_opts: Option<HashMap<String, PyRunnerOpt>>,
    model_name: Option<String>,
    short_description: Option<String>,
    model_description: Option<String>,
    required_platforms: Option<Vec<String>>,
    inputs: Option<Vec<TensorSpec>>,
    outputs: Option<Vec<TensorSpec>>,
    self_tests: Option<Vec<SelfTest>>,
    examples: Option<Vec<Example>>,
    misc_files: Option<HashMap<String, Vec<u8>>>,
    linked_files: Option<HashMap<String, Vec<String>>>,
) -> PyResult<carton_core::types::PackOpts<PyTensorStorage>> {
    let misc_files: Option<HashMap<String, LazyLoadedMiscFile>> = convert_opt_map(misc_files);

    Ok(carton_core::types::PackOpts {
        info: carton_core::types::CartonInfo {
            model_name,
            short_description,
            model_description,
            required_platforms: convert_required_platforms(required_platforms)?,
            inputs: convert_opt_vec(inputs),
            outputs: convert_opt_vec(outputs),
            self_tests: convert_opt_vec(self_tests),
            examples: convert_opt_vec(examples),
            runner: carton_core::info::RunnerInfo {
                runner_name,
                required_framework_version: VersionReq::from_str(&required_framework_version)
                    .map_err(|e| {
                        PyValueError::new_err(format!("Invalid `required_framework_version`: {e}"))
                    })?,
                runner_compat_version,
                opts: convert_opt_map(runner_opts),
            },
            misc_files: convert_opt_map(misc_files),
        },
        linked_files: linked_files.map(|v| {
            v.into_iter()
                .map(|(k, v)| LinkedFile { sha256: k, urls: v })
                .collect()
        }),
    })
}

// Info about a carton
#[pyclass]
#[derive(FromPyObject, Debug)]
pub(crate) struct CartonInfo {
    /// The name of the model
    #[pyo3(get)]
    pub model_name: Option<String>,

    /// A short description (should be 100 characters or less)
    #[pyo3(get)]
    pub short_description: Option<String>,

    /// The model description
    #[pyo3(get)]
    pub model_description: Option<String>,

    /// A list of platforms this model supports
    /// If empty or unspecified, all platforms are okay
    #[pyo3(get)]
    pub required_platforms: Option<Vec<String>>,

    /// A list of inputs for the model
    /// Can be empty
    #[pyo3(get)]
    pub inputs: Option<Vec<TensorSpec>>,

    /// A list of outputs for the model
    /// Can be empty
    #[pyo3(get)]
    pub outputs: Option<Vec<TensorSpec>>,

    /// Test data
    /// Can be empty
    #[pyo3(get)]
    pub self_tests: Option<Vec<SelfTest>>,

    /// Examples
    /// Can be empty
    #[pyo3(get)]
    pub examples: Option<Vec<Example>>,

    /// Information about the runner to use
    #[pyo3(get)]
    pub runner: RunnerInfo,

    /// Misc files that can be referenced by the description. The key is a normalized relative path
    /// (i.e one that does not reference parent directories, etc)
    #[pyo3(get)]
    pub misc_files: Option<HashMap<String, LazyLoadedMiscFile>>,
}

#[pymethods]
impl CartonInfo {
    fn __str__(&self) -> String {
        format!("{self:#?}")
    }
}

fn convert_required_platforms(r: Option<Vec<String>>) -> PyResult<Option<Vec<Triple>>> {
    Ok(match r {
        Some(required_platforms) => {
            let mut out = Vec::new();
            for plat in required_platforms {
                out.push(Triple::from_str(&plat).map_err(|e| {
                    PyValueError::new_err(format!("Invalid value in `required_platforms`: {e}"))
                })?)
            }

            Some(out)
        }
        None => None,
    })
}

impl From<carton_core::types::CartonInfo<carton_core::types::GenericStorage>> for CartonInfo {
    fn from(value: carton_core::types::CartonInfo<carton_core::types::GenericStorage>) -> Self {
        Self {
            model_name: value.model_name,
            short_description: value.short_description,
            model_description: value.model_description,
            required_platforms: value.required_platforms.map(|required_platforms| {
                required_platforms
                    .into_iter()
                    .map(|plat| plat.to_string())
                    .collect()
            }),
            inputs: convert_opt_vec(value.inputs),
            outputs: convert_opt_vec(value.outputs),
            self_tests: convert_opt_vec(value.self_tests),
            examples: convert_opt_vec(value.examples),
            runner: RunnerInfo {
                runner_name: value.runner.runner_name,
                required_framework_version: value.runner.required_framework_version.to_string(),
                runner_compat_version: value.runner.runner_compat_version,
                opts: convert_opt_map(value.runner.opts),
            },
            misc_files: convert_opt_map(value.misc_files),
        }
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub(crate) struct TensorSpec {
    #[pyo3(get, set)]
    pub name: String,

    // Note: getter and setter implemented below
    /// The datatype
    pub dtype: DataType,

    /// Tensor shape
    #[pyo3(get, set)]
    pub shape: Shape,

    /// Optional description
    #[pyo3(get, set)]
    pub description: Option<String>,

    /// Optional internal name
    #[pyo3(get, set)]
    pub internal_name: Option<String>,
}

#[pymethods]
impl TensorSpec {
    #[getter]
    fn get_dtype(&self) -> String {
        self.dtype.to_str().to_owned()
    }

    #[setter]
    fn set_dtype(&mut self, value: &str) -> PyResult<()> {
        self.dtype = DataType::from_str(value).map_err(|e| PyValueError::new_err(e))?;

        Ok(())
    }

    #[new]
    fn new<'py>(
        name: String,
        dtype: &str,
        shape: Shape,
        description: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            name,
            dtype: DataType::from_str(dtype).map_err(|e| PyValueError::new_err(e))?,
            shape,
            description,
            internal_name: None,
        })
    }
}

impl From<TensorSpec> for carton_core::info::TensorSpec {
    fn from(value: TensorSpec) -> Self {
        Self {
            name: value.name,
            dtype: value.dtype,
            shape: value.shape.into(),
            description: value.description,
            internal_name: value.internal_name,
        }
    }
}

impl From<carton_core::info::TensorSpec> for TensorSpec {
    fn from(value: carton_core::info::TensorSpec) -> Self {
        Self {
            name: value.name,
            dtype: value.dtype,
            shape: value.shape.into(),
            description: value.description,
            internal_name: value.internal_name,
        }
    }
}

#[derive(Clone, FromPyObject, Debug)]
pub enum Shape {
    /// Any shape
    Any(#[pyo3(from_py_with = "handle_none")] ()),

    /// A symbol for the whole shape
    Symbol(String),

    /// A list of dimensions
    /// An empty vec is considered a scalar
    Shape(Vec<Dimension>),
}

impl From<Shape> for carton_core::info::Shape {
    fn from(value: Shape) -> Self {
        match value {
            Shape::Any(_) => Self::Any,
            Shape::Symbol(v) => Self::Symbol(v),
            Shape::Shape(v) => Self::Shape(convert_vec(v)),
        }
    }
}

impl From<carton_core::info::Shape> for Shape {
    fn from(value: carton_core::info::Shape) -> Self {
        match value {
            carton_core::info::Shape::Any => Self::Any(()),
            carton_core::info::Shape::Symbol(v) => Self::Symbol(v),
            carton_core::info::Shape::Shape(v) => Self::Shape(convert_vec(v)),
        }
    }
}

impl IntoPy<PyObject> for Shape {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            Shape::Any(_) => Python::None(py),
            Shape::Symbol(item) => item.into_py(py),
            Shape::Shape(item) => item.into_py(py),
        }
    }
}

fn handle_none(item: &PyAny) -> PyResult<()> {
    if item.is_none() {
        Ok(())
    } else {
        Err(PyDowncastError::new(item, "None").into())
    }
}

/// A dimension can be either a fixed value, a symbol, or any value
#[derive(Clone, FromPyObject, Debug)]
pub enum Dimension {
    Value(u64),
    Symbol(String),
    Any(#[pyo3(from_py_with = "handle_none")] ()),
}

impl From<Dimension> for carton_core::info::Dimension {
    fn from(value: Dimension) -> Self {
        match value {
            Dimension::Value(v) => Self::Value(v),
            Dimension::Symbol(v) => Self::Symbol(v),
            Dimension::Any(_) => Self::Any,
        }
    }
}

impl From<carton_core::info::Dimension> for Dimension {
    fn from(value: carton_core::info::Dimension) -> Self {
        match value {
            carton_core::info::Dimension::Value(v) => Self::Value(v),
            carton_core::info::Dimension::Symbol(v) => Self::Symbol(v),
            carton_core::info::Dimension::Any => Self::Any(()),
        }
    }
}

impl IntoPy<PyObject> for Dimension {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            Dimension::Value(item) => item.into_py(py),
            Dimension::Symbol(item) => item.into_py(py),
            Dimension::Any(_) => Python::None(py),
        }
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub(crate) struct SelfTest {
    #[pyo3(get, set)]
    pub name: Option<String>,

    #[pyo3(get, set)]
    pub description: Option<String>,

    #[pyo3(get, set)]
    pub inputs: HashMap<String, LazyLoadedTensor>,

    // Can be empty
    #[pyo3(get, set)]
    pub expected_out: Option<HashMap<String, LazyLoadedTensor>>,
}

#[pymethods]
impl SelfTest {
    #[new]
    fn new<'py>(
        inputs: HashMap<String, SupportedTensorType<'py>>,
        name: Option<String>,
        description: Option<String>,
        expected_out: Option<HashMap<String, SupportedTensorType<'py>>>,
    ) -> PyResult<Self> {
        Ok(Self {
            name,
            description,
            inputs: convert_map(inputs),
            expected_out: convert_opt_map(expected_out),
        })
    }
}

impl From<SelfTest> for carton_core::info::SelfTest<PyTensorStorage> {
    fn from(value: SelfTest) -> Self {
        Self {
            name: value.name,
            description: value.description,
            inputs: convert_map(value.inputs),
            expected_out: convert_opt_map(value.expected_out),
        }
    }
}

impl From<carton_core::info::SelfTest<carton_core::types::GenericStorage>> for SelfTest {
    fn from(value: carton_core::info::SelfTest<carton_core::types::GenericStorage>) -> Self {
        Self {
            name: value.name,
            description: value.description,
            inputs: convert_map(value.inputs),
            expected_out: convert_opt_map(value.expected_out),
        }
    }
}

#[derive(FromPyObject)]
pub(crate) enum PyArrayOrMisc<'py> {
    Tensor(SupportedTensorType<'py>),
    Misc(Vec<u8>),
}

impl<'py> From<PyArrayOrMisc<'py>> for TensorOrMisc {
    fn from(value: PyArrayOrMisc<'py>) -> Self {
        match value {
            PyArrayOrMisc::Tensor(v) => Self::Tensor(v.into()),
            PyArrayOrMisc::Misc(v) => Self::Misc(v.into()),
        }
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub(crate) struct Example {
    #[pyo3(get, set)]
    pub name: Option<String>,

    #[pyo3(get, set)]
    pub description: Option<String>,

    #[pyo3(get, set)]
    pub inputs: HashMap<String, TensorOrMisc>,

    #[pyo3(get, set)]
    pub sample_out: HashMap<String, TensorOrMisc>,
}

#[pymethods]
impl Example {
    #[new]
    fn new<'py>(
        inputs: HashMap<String, PyArrayOrMisc<'py>>,
        sample_out: HashMap<String, PyArrayOrMisc<'py>>,
        name: Option<String>,
        description: Option<String>,
    ) -> PyResult<Self> {
        Ok(Self {
            name,
            description,
            inputs: convert_map(inputs),
            sample_out: convert_map(sample_out),
        })
    }
}

impl From<Example> for carton_core::info::Example<PyTensorStorage> {
    fn from(value: Example) -> Self {
        Self {
            name: value.name,
            description: value.description,
            inputs: convert_map(value.inputs),
            sample_out: convert_map(value.sample_out),
        }
    }
}

impl From<carton_core::info::Example<GenericStorage>> for Example {
    fn from(value: carton_core::info::Example<GenericStorage>) -> Self {
        Self {
            name: value.name,
            description: value.description,
            inputs: convert_map(value.inputs),
            sample_out: convert_map(value.sample_out),
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub(crate) struct LazyLoadedTensor {
    inner: carton_core::info::PossiblyLoaded<Tensor<PyTensorStorage>>,
}

impl std::fmt::Debug for LazyLoadedTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazyLoadedTensor").finish()
    }
}

impl<'py> From<SupportedTensorType<'py>> for LazyLoadedTensor {
    fn from(value: SupportedTensorType<'py>) -> Self {
        LazyLoadedTensor {
            inner: carton_core::info::PossiblyLoaded::from_value(value.into()),
        }
    }
}

#[pymethods]
impl LazyLoadedTensor {
    fn get<'a>(&self, py: Python<'a>) -> PyResult<&'a PyAny> {
        let inner = self.inner.clone();
        pyo3_asyncio::tokio::future_into_py(py, async move {
            let t = inner.get().await;
            Ok(tensor_to_py(t))
        })
    }
}

impl From<LazyLoadedTensor> for carton_core::info::PossiblyLoaded<Tensor<PyTensorStorage>> {
    fn from(value: LazyLoadedTensor) -> Self {
        value.inner
    }
}

impl From<carton_core::info::PossiblyLoaded<carton_core::types::Tensor<GenericStorage>>>
    for LazyLoadedTensor
{
    fn from(
        value: carton_core::info::PossiblyLoaded<carton_core::types::Tensor<GenericStorage>>,
    ) -> Self {
        Self {
            inner: carton_core::info::PossiblyLoaded::from_loader(Box::pin(async move {
                let item = value.get().await;

                // TODO: this makes a copy
                Python::with_gil(|py| {
                    match item {
                        Tensor::Float(item) => Tensor::Float(item.view().to_pyarray(py).into()),
                        Tensor::Double(item) => Tensor::Double(item.view().to_pyarray(py).into()),
                        // Tensor::String(item) => item.view().to_pyarray(py).to_object(py),
                        Tensor::String(_) => panic!("String tensor output not implemented yet"),
                        Tensor::I8(item) => Tensor::I8(item.view().to_pyarray(py).into()),
                        Tensor::I16(item) => Tensor::I16(item.view().to_pyarray(py).into()),
                        Tensor::I32(item) => Tensor::I32(item.view().to_pyarray(py).into()),
                        Tensor::I64(item) => Tensor::I64(item.view().to_pyarray(py).into()),
                        Tensor::U8(item) => Tensor::U8(item.view().to_pyarray(py).into()),
                        Tensor::U16(item) => Tensor::U16(item.view().to_pyarray(py).into()),
                        Tensor::U32(item) => Tensor::U32(item.view().to_pyarray(py).into()),
                        Tensor::U64(item) => Tensor::U64(item.view().to_pyarray(py).into()),
                        Tensor::NestedTensor(_) => {
                            panic!("Nested tensor output not implemented yet")
                        }
                    }
                })
            })),
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub(crate) struct LazyLoadedMiscFile {
    inner: Arc<LazyLoadedMiscFileInner>,
}

struct LazyLoadedMiscFileInner {
    loader: carton_core::info::ArcMiscFileLoader,

    // Once the file has been loaded
    // TODO: see if we can avoid a mutex here
    item: Mutex<Option<carton_core::info::MiscFile>>,
}

impl std::fmt::Debug for LazyLoadedMiscFile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LazyLoadedMiscFile").finish()
    }
}

/// An in memory misc file
#[derive(Clone)]
struct PyMiscFile(Arc<Vec<u8>>);

impl AsRef<[u8]> for PyMiscFile {
    fn as_ref(&self) -> &[u8] {
        self.0.as_ref()
    }
}

struct PyMiscFileLoader {
    data: PyMiscFile,
}

#[async_trait]
impl carton_core::info::MiscFileLoader for PyMiscFileLoader {
    async fn get(&self) -> carton_core::info::MiscFile {
        Box::new(std::io::Cursor::new(self.data.clone()))
    }
}

impl From<Vec<u8>> for LazyLoadedMiscFile {
    fn from(value: Vec<u8>) -> Self {
        LazyLoadedMiscFile {
            inner: Arc::new(LazyLoadedMiscFileInner {
                loader: Arc::new(PyMiscFileLoader {
                    data: PyMiscFile(Arc::new(value)),
                }),
                item: Mutex::new(None),
            }),
        }
    }
}

#[pymethods]
impl LazyLoadedMiscFile {
    fn read<'a>(&self, py: Python<'a>, size_bytes: Option<u64>) -> PyResult<&'a PyAny> {
        let inner = self.inner.clone();
        pyo3_asyncio::tokio::future_into_py(py, async move {
            // Actually load the file if we need to
            let mut item = inner.item.lock().await;
            if item.is_none() {
                *item = Some(inner.loader.get().await);
            }

            if let Some(file) = item.as_mut() {
                let buf = if let Some(size_bytes) = size_bytes {
                    let mut buf = vec![0; size_bytes as usize];
                    file.read_exact(&mut buf).await.unwrap();
                    buf
                } else {
                    let mut buf = Vec::new();
                    file.read_to_end(&mut buf).await.unwrap();
                    buf
                };

                // TODO: this makes a copy
                let out: PyObject = Python::with_gil(|py| PyBytes::new(py, &buf).into());
                Ok(out)
            } else {
                panic!("Error when loading file from carton. Please file an issue.")
            }
        })
    }
}

impl From<LazyLoadedMiscFile> for carton_core::info::ArcMiscFileLoader {
    fn from(value: LazyLoadedMiscFile) -> Self {
        value.inner.loader.clone()
    }
}

impl From<carton_core::info::ArcMiscFileLoader> for LazyLoadedMiscFile {
    fn from(value: carton_core::info::ArcMiscFileLoader) -> Self {
        Self {
            inner: Arc::new(LazyLoadedMiscFileInner {
                loader: value,
                item: Mutex::new(None),
            }),
        }
    }
}

#[derive(Clone, FromPyObject, Debug)]
pub(crate) enum TensorOrMisc {
    Tensor(LazyLoadedTensor),
    Misc(LazyLoadedMiscFile),
}

impl From<TensorOrMisc> for carton_core::info::TensorOrMisc<PyTensorStorage> {
    fn from(value: TensorOrMisc) -> Self {
        match value {
            TensorOrMisc::Tensor(v) => Self::Tensor(v.into()),
            TensorOrMisc::Misc(v) => Self::Misc(v.into()),
        }
    }
}

impl From<carton_core::info::TensorOrMisc<GenericStorage>> for TensorOrMisc {
    fn from(value: carton_core::info::TensorOrMisc<GenericStorage>) -> Self {
        match value {
            carton_core::info::TensorOrMisc::Tensor(v) => Self::Tensor(v.into()),
            carton_core::info::TensorOrMisc::Misc(v) => Self::Misc(v.into()),
        }
    }
}

impl IntoPy<PyObject> for TensorOrMisc {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            TensorOrMisc::Tensor(v) => v.into_py(py),
            TensorOrMisc::Misc(v) => v.into_py(py),
        }
    }
}

#[pyclass]
#[derive(Clone, Debug)]
pub(crate) struct RunnerInfo {
    /// The name of the runner to use
    #[pyo3(get)]
    pub runner_name: String,

    /// The required framework version range to run the model with
    /// This is a semver version range. See https://docs.rs/semver/1.0.16/semver/struct.VersionReq.html
    /// for format details.
    /// For example `=1.2.4`, means exactly version `1.2.4`
    /// In most cases, this should be exactly one version
    #[pyo3(get)]
    pub required_framework_version: String,

    /// Don't set this unless you know what you're doing
    #[pyo3(get)]
    pub runner_compat_version: Option<u64>,

    /// Options to pass to the runner. These are runner-specific (e.g.
    /// PyTorch, TensorFlow, etc).
    ///
    /// Sometimes used to configure thread-pool sizes, etc.
    /// See the documentation for more info
    #[pyo3(get)]
    pub opts: Option<HashMap<String, PyRunnerOpt>>,
}

#[derive(FromPyObject, Clone, Debug)]
pub(crate) enum PyRunnerOpt {
    Integer(i64),
    Double(f64),
    String(String),
    Boolean(bool),
}

impl From<PyRunnerOpt> for RunnerOpt {
    fn from(value: PyRunnerOpt) -> Self {
        match value {
            PyRunnerOpt::Integer(v) => Self::Integer(v),
            PyRunnerOpt::Double(v) => Self::Double(v),
            PyRunnerOpt::String(v) => Self::String(v),
            PyRunnerOpt::Boolean(v) => Self::Boolean(v),
        }
    }
}

impl From<RunnerOpt> for PyRunnerOpt {
    fn from(value: RunnerOpt) -> Self {
        match value {
            RunnerOpt::Integer(v) => Self::Integer(v),
            RunnerOpt::Double(v) => Self::Double(v),
            RunnerOpt::String(v) => Self::String(v),
            RunnerOpt::Boolean(v) => Self::Boolean(v),
        }
    }
}

impl IntoPy<PyObject> for PyRunnerOpt {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            PyRunnerOpt::Integer(v) => v.into_py(py),
            PyRunnerOpt::Double(v) => v.into_py(py),
            PyRunnerOpt::String(v) => v.into_py(py),
            PyRunnerOpt::Boolean(v) => v.into_py(py),
        }
    }
}
