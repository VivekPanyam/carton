use std::{collections::HashMap, sync::Arc};

use conversions::{
    create_load_opts, create_pack_opts, CartonInfo, Device, Example, LazyLoadedMiscFile,
    LazyLoadedTensor, PyRunnerOpt, RunnerInfo, SelfTest, TensorSpec,
};
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
use tensor::{tensor_to_py, SupportedTensorType};

mod conversions;
mod tensor;

#[pyclass]
#[derive(Clone)]
struct SealHandle {
    inner: carton_core::types::SealHandle,
}

#[pyclass]
struct Carton {
    inner: Arc<carton_core::Carton>,
}

#[pymethods]
impl Carton {
    fn infer<'a>(&self, py: Python<'a>, tensors: &PyDict) -> PyResult<&'a PyAny> {
        let tensors: HashMap<String, SupportedTensorType> = tensors.extract().unwrap();
        let transformed = tensors.into_iter().map(|(k, v)| (k, v.into())).collect();

        let inner = self.inner.clone();
        pyo3_asyncio::tokio::future_into_py(py, async move {
            let out: HashMap<String, PyObject> = inner
                .infer_with_inputs(transformed)
                .await
                .unwrap()
                .into_iter()
                .map(|(k, v)| (k, tensor_to_py(&v)))
                .collect();

            Ok(out)
        })
    }

    fn seal<'a>(&self, py: Python<'a>, tensors: &PyDict) -> PyResult<&'a PyAny> {
        let tensors: HashMap<String, SupportedTensorType> = tensors.extract().unwrap();
        let transformed = tensors.into_iter().map(|(k, v)| (k, v.into())).collect();

        let inner = self.inner.clone();
        pyo3_asyncio::tokio::future_into_py(py, async move {
            let out = inner.seal(transformed).await.unwrap();
            Ok(SealHandle { inner: out })
        })
    }

    fn infer_with_handle<'a>(&self, py: Python<'a>, handle: SealHandle) -> PyResult<&'a PyAny> {
        let inner = self.inner.clone();
        pyo3_asyncio::tokio::future_into_py(py, async move {
            let out: HashMap<String, PyObject> = inner
                .infer_with_handle(handle.inner)
                .await
                .unwrap()
                .into_iter()
                .map(|(k, v)| (k, tensor_to_py(&v)))
                .collect();

            Ok(out)
        })
    }

    #[getter]
    fn info(&self) -> CartonInfo {
        // TODO: maybe cache this conversion?
        (*self.inner.get_info()).info.clone().into()
    }
}

/// Load a model
#[pyfunction]
fn load(
    py: Python,
    path: String,
    visible_device: Option<Device>,
    override_runner_name: Option<String>,
    override_required_framework_version: Option<String>,
    override_runner_opts: Option<HashMap<String, PyRunnerOpt>>,
) -> PyResult<&PyAny> {
    pyo3_asyncio::tokio::future_into_py(py, async move {
        let opts = create_load_opts(
            visible_device,
            override_runner_name,
            override_required_framework_version,
            override_runner_opts,
        )?;

        // TODO: use something more specific than ValueError
        let inner = carton_core::Carton::load(path, opts)
            .await
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Carton {
            inner: Arc::new(inner),
        })
    })
}

/// Load an unpacked model
/// Has all the options of `pack` and the non-override options of `load`
#[pyfunction]
fn load_unpacked(
    py: Python,
    path: String,
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
    visible_device: Option<Device>,
) -> PyResult<&PyAny> {
    pyo3_asyncio::tokio::future_into_py(py, async move {
        let pack_opts = create_pack_opts(
            runner_name,
            required_framework_version,
            runner_compat_version,
            runner_opts,
            model_name,
            short_description,
            model_description,
            required_platforms,
            inputs,
            outputs,
            self_tests,
            examples,
            misc_files,
        )?;

        // No need for overrides here
        let load_opts = create_load_opts(visible_device, None, None, None)?;

        let inner = carton_core::Carton::load_unpacked(path, pack_opts, load_opts)
            .await
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(Carton {
            inner: Arc::new(inner),
        })
    })
}

/// Pack a model
#[pyfunction]
fn pack(
    py: Python,
    path: String,
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
) -> PyResult<&PyAny> {
    pyo3_asyncio::tokio::future_into_py(py, async move {
        let opts = create_pack_opts(
            runner_name,
            required_framework_version,
            runner_compat_version,
            runner_opts,
            model_name,
            short_description,
            model_description,
            required_platforms,
            inputs,
            outputs,
            self_tests,
            examples,
            misc_files,
        )?;

        let out = carton_core::Carton::pack(path, opts)
            .await
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(out)
    })
}

/// Get info for a model
#[pyfunction]
fn get_model_info(py: Python, url_or_path: String) -> PyResult<&PyAny> {
    pyo3_asyncio::tokio::future_into_py(py, async move {
        let out: CartonInfo = carton_core::Carton::get_model_info(url_or_path)
            .await
            .map(|v| v.info)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .into();

        Ok(out)
    })
}

/// Shrink a packed carton by storing links to files instead of the files themselves when possible.
/// Takes a path to a packed carton along with a mapping from sha256 to a list of URLs
/// Returns the path to another packed carton
#[pyfunction]
fn shrink(
    py: Python,
    path: std::path::PathBuf,
    urls: HashMap<String, Vec<String>>,
) -> PyResult<&PyAny> {
    pyo3_asyncio::tokio::future_into_py(py, async move {
        Ok(carton_core::Carton::shrink(path, urls)
            .await
            .map_err(|e| PyValueError::new_err(e.to_string()))?)
    })
}

/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn cartonml(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(pack, m)?)?;
    m.add_function(wrap_pyfunction!(load_unpacked, m)?)?;
    m.add_function(wrap_pyfunction!(get_model_info, m)?)?;
    m.add_function(wrap_pyfunction!(shrink, m)?)?;
    m.add_class::<Carton>()?;
    m.add_class::<CartonInfo>()?;
    m.add_class::<TensorSpec>()?;
    m.add_class::<SelfTest>()?;
    m.add_class::<Example>()?;
    m.add_class::<LazyLoadedTensor>()?;
    m.add_class::<LazyLoadedMiscFile>()?;
    m.add_class::<RunnerInfo>()?;
    Ok(())
}
