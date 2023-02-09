use std::{ffi::OsStr, path::Path};

use pyo3::{PyResult, Python};

pub(crate) fn get_executable_path() -> PyResult<String> {
    Python::with_gil(|py| Python::import(py, "sys")?.getattr("executable")?.extract())
}

/// Adds a vec of paths to sys.path
/// Also updates the PYTHONPATH env var
pub(crate) fn add_to_sys_path<P: AsRef<Path>>(paths: &Vec<P>) -> pyo3::PyResult<()> {
    Python::with_gil(|py| {
        let sys_path_insert = Python::import(py, "sys")?
            .getattr("path")?
            .getattr("insert")?;

        for item in paths {
            sys_path_insert.call1((0, item.as_ref()))?;
        }

        // TODO: PYTHONPATH is not inserted at the beginning of sys.path in new processes. This might cause
        // problems if a package we want to use is also available in the built in installation
        let to_add = paths
            .into_iter()
            .map(|p| p.as_ref().as_os_str())
            .collect::<Vec<_>>()
            .join(OsStr::new(":"));

        // There's a race condition between reading and writing PYTHONPATH, but it shouldn't generally matter
        let new_path =
            to_add.into_string().unwrap() + ":" + &std::env::var("PYTHONPATH").unwrap_or_default();
        std::env::set_var("PYTHONPATH", new_path);

        Ok(())
    })
}
