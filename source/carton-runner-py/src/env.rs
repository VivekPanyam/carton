use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, FromPyObject, PartialEq, Default)]
pub struct EnvironmentMarkers {
    // List of markers from  https://peps.python.org/pep-0508/#environment-markers
    pub os_name: Option<String>,
    pub sys_platform: Option<String>,
    pub platform_machine: Option<String>,
    pub platform_python_implementation: Option<String>,
    pub platform_release: Option<String>,
    pub platform_system: Option<String>,
    pub python_version: Option<String>,
    pub python_full_version: Option<String>,
    pub implementation_name: Option<String>,
    pub implementation_version: Option<String>,
}

impl EnvironmentMarkers {
    pub fn get_current() -> PyResult<EnvironmentMarkers> {
        Python::with_gil(|py| {
            let fun = PyModule::from_code(
                py,
                r#"
import os
import sys
import platform

class EnvironmentMarkers:
    """
    All this code is based on https://peps.python.org/pep-0508/#environment-markers
    """
    def __init__(self):
        self.os_name = os.name
        self.sys_platform = sys.platform
        self.platform_machine = platform.machine()
        self.platform_python_implementation = platform.python_implementation()
        self.platform_release = platform.release()
        self.platform_system = platform.system()
        self.python_version = '.'.join(platform.python_version_tuple()[:2])
        self.python_full_version = platform.python_version()
        self.implementation_name = sys.implementation.name
        self.implementation_version = get_implementation_version()

def format_full_version(info):
    version = '{0.major}.{0.minor}.{0.micro}'.format(info)
    kind = info.releaselevel
    if kind != 'final':
        version += kind[0] + str(info.serial)
    return version

def get_implementation_version():
    if hasattr(sys, 'implementation'):
        return format_full_version(sys.implementation.version)
    else:
        return "0"
"#,
                "",
                "",
            )?
            .getattr("EnvironmentMarkers")?;

            let out: EnvironmentMarkers = fun.call0()?.extract()?;

            Ok(out)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::EnvironmentMarkers;

    #[test]
    fn get_current_environment() {
        crate::python_utils::init();

        let env = EnvironmentMarkers::get_current().unwrap();
        println!("{:#?}", env);
    }

    #[test]
    fn serailize_empty() {
        crate::python_utils::init();

        let env = EnvironmentMarkers::default();
        let serialized = toml::to_string_pretty(&env).unwrap();
        assert!(serialized.is_empty())
    }
}
