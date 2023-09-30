use carton_core::error::CartonError;
use carton_core::types::{Device, LoadOpts};
use carton_core::Carton;
use ocaml_rust::Custom;
use tokio::runtime::Runtime;

#[ocaml_rust::bridge]
mod ffi {
    type FFICarton = Custom<Carton>;
    type FFICartonError = Custom<CartonError>;
    extern "Rust" {
        fn load(
            path: String,
            visible_device: Option<String>,
            override_runner_name: Option<String>,
            override_required_framework_version: Option<String>,
        ) -> Result<FFICarton, FFICartonError>;
    }
}

/// Load a carton model
fn load(
    path: String,
    visible_device: Option<String>,
    override_runner_name: Option<String>,
    override_required_framework_version: Option<String>,
) -> Result<Custom<Carton>, Custom<CartonError>> {
    Runtime::new().unwrap().block_on(async {
        let opts = LoadOpts {
            override_runner_name,
            override_required_framework_version,
            override_runner_opts: None,
            visible_device: match visible_device {
                None => carton_core::types::Device::default(),
                Some(visible_device) => Device::maybe_from_str(&visible_device).unwrap(),
            },
        };

        match Carton::load(path, opts).await {
            Ok(x) => Ok(Custom::new(x)),
            Err(x) => Err(Custom::new(x)),
        }
    })
}
