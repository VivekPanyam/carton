use findshlibs::{SharedLibrary, TargetSharedLibrary};
use pyo3::{PyResult, Python};
use std::{ffi::OsStr, path::Path, sync::Once};

/// Initialize python with an isolated environment
/// This must be called before attempting to use python (otherwise any PyO3 code will panic)
/// (safe to call multiple times)
pub(crate) fn init() {
    static INIT: Once = Once::new();
    INIT.call_once(|| init_inner())
}

fn init_inner() {
    // Explicitly ignore pythonpath because we want to run in an isolated environment
    std::env::remove_var("PYTHONPATH");

    // Isolate
    std::env::set_var("PYTHONNOUSERSITE", "true");

    // Get the loaded python library
    let mut pythonlib = None;
    TargetSharedLibrary::each(|shlib| {
        let name = shlib.name().to_string_lossy();
        log::trace!("Found library in memory: {name}");

        if name.contains("libpython") {
            if pythonlib.is_some() {
                panic!("Found multiple python libraries loaded during python runner startup!");
            }

            pythonlib = Some(name.to_string());
        }
    });

    let pythonlib = match pythonlib {
        Some(s) => s,
        None => panic!("Didn't find libpython in memory during python runner startup. This shouldn't happen so please file a github issue.")
    };

    // TODO: make this check more robust. Maybe look for a specific marker file?
    if !pythonlib.contains("bundled_python") {
        panic!("It seems like we're not using the bundled python interpreter! Found '{pythonlib}', but expected a path containing 'bundled_python'")
    }

    // Set PYTHONHOME
    let pythonhome = std::path::Path::new(&pythonlib)
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    log::trace!("Setting PYTHONHOME to {pythonhome:?}");
    std::env::set_var("PYTHONHOME", pythonhome);

    // Start the interpreter
    pyo3::prepare_freethreaded_python();

    // Set sys.executable
    Python::with_gil(|py| {
        let info = py.version_info();
        let executable = pythonhome.join(format!("bin/python{}.{}", info.major, info.minor));
        log::trace!("Setting sys.executable to {executable:?}");
        py.import("sys")
            .unwrap()
            .setattr("executable", executable.as_os_str())
            .unwrap();

        // Only does anything on mac
        // Note: it's okay to set this after interpreter initialization because this is only used by subprocesses
        std::env::set_var("PYTHONEXECUTABLE", "");
    });
}

#[cfg(not(all(test, target_os = "macos")))]
pub(crate) fn get_executable_path() -> PyResult<String> {
    Python::with_gil(|py| Python::import(py, "sys")?.getattr("executable")?.extract())
}

// TODO: make this more generic
#[cfg(all(test, target_os = "macos"))]
pub(crate) fn get_executable_path() -> PyResult<String> {
    Ok("/Applications/Xcode.app/Contents/Developer/usr/bin/python3".into())
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
