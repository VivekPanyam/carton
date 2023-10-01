mod export;
mod tensor;

use crate::export::Tensor;

use carton_core::error::CartonError;
use carton_core::types::{Device, LoadOpts, Tensor as CartonTensor};
use carton_core::Carton;
use ocaml_rust::{Custom, CustomConst};
use tokio::runtime::Runtime;

/// Load a carton model
fn load(
    path: String,
    visible_device: Option<String>,
    override_runner_name: Option<String>,
    override_required_framework_version: Option<String>,
) -> Result<CustomConst<Carton>, Custom<CartonError>> {
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
            Ok(x) => Ok(CustomConst::new(x)),
            Err(x) => Err(Custom::new(x)),
        }
    })
}

/// Infer the data
fn infer(model: CustomConst<Carton>, tensors: Vec<(String, Tensor)>) -> Vec<(String, Tensor)> {
    Runtime::new().unwrap().block_on(async {
        let transformed: Vec<(_, CartonTensor<_>)> = tensors
            .into_iter()
            .map(|(k, v)| (k, CartonTensor::from(v)))
            .collect();

        model
            .inner()
            .infer(transformed)
            .await
            .unwrap()
            .into_iter()
            .map(|(x, y)| (x, y.into()))
            .collect()
    })
}
