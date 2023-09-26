// Copyright 2023 Vivek Panyam
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::collections::HashMap;

use wasm_bindgen::prelude::*;

use carton_core::{
    conversion_utils::convert_vec,
    info::{ArcMiscFileLoader, PossiblyLoaded},
    types::{for_each_numeric_carton_type, GenericStorage, Tensor},
};
use serde::ser::Serialize;
use tokio_util::compat::TokioAsyncReadCompatExt;
use wasm_streams::ReadableStream;

mod utils;

pub struct CartonError(carton_core::error::CartonError);

impl From<CartonError> for JsValue {
    fn from(value: CartonError) -> Self {
        value.0.to_string().into()
    }
}

impl From<carton_core::error::CartonError> for CartonError {
    fn from(value: carton_core::error::CartonError) -> Self {
        Self(value)
    }
}

#[wasm_bindgen]
pub async fn get_model_info(url: String) -> Result<CartonInfo, CartonError> {
    // TODO: we want to call this from all possible entrypoints (including registration code)
    utils::init_logging();
    let info = carton_core::Carton::get_model_info(url).await;

    info.map(|v| v.into()).map_err(|e| e.into())
}

// Info about a carton
#[wasm_bindgen(getter_with_clone)]
pub struct CartonInfo {
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
    pub required_platforms: JsValue,

    /// A list of inputs for the model
    /// Can be empty
    pub inputs: JsValue,

    /// A list of outputs for the model
    /// Can be empty
    pub outputs: JsValue,

    /// Test data
    /// Can be empty
    pub self_tests: Option<js_sys::Array>, // Vec<SelfTest>,

    /// Examples
    /// Can be empty
    pub examples: Option<js_sys::Array>, // Vec<Example>,

    /// Information about the runner to use
    pub runner: JsValue,

    /// Misc files that can be referenced by the description. The key is a normalized relative path
    /// (i.e one that does not reference parent directories, etc)
    pub misc_files: Option<js_sys::Map>, // HashMap<String, MiscFileLoaderWrapper>

    /// The sha256 of the MANIFEST file (if available)
    /// This should always be available unless we're running an unpacked model
    // Note: this field is not directly in CartonInfo in the rust library
    pub manifest_sha256: Option<String>,
}

impl From<carton_core::info::CartonInfoWithExtras<GenericStorage>> for CartonInfo {
    fn from(info: carton_core::info::CartonInfoWithExtras<GenericStorage>) -> Self {
        let value = info.info;
        let serializer = serde_wasm_bindgen::Serializer::json_compatible();
        Self {
            model_name: value.model_name,
            short_description: value.short_description,
            model_description: value.model_description,
            license: value.license,
            repository: value.repository,
            homepage: value.homepage,
            required_platforms: value.required_platforms.serialize(&serializer).unwrap(),
            inputs: value.inputs.serialize(&serializer).unwrap(),
            outputs: value.outputs.serialize(&serializer).unwrap(),
            self_tests: value.self_tests.map(|v| {
                convert_vec::<_, SelfTest>(v)
                    .into_iter()
                    .map(JsValue::from)
                    .collect()
            }),
            examples: value.examples.map(|v| {
                convert_vec::<_, Example>(v)
                    .into_iter()
                    .map(JsValue::from)
                    .collect()
            }),
            runner: value.runner.serialize(&serializer).unwrap(),
            misc_files: value.misc_files.map(|v| {
                to_map(
                    v.into_iter()
                        .map(|(k, v)| (k, MiscFileLoaderWrapper(v)))
                        .collect(),
                )
            }),
            manifest_sha256: info.manifest_sha256,
        }
    }
}

fn to_map<K: Into<JsValue>, V: Into<JsValue>>(value: HashMap<K, V>) -> js_sys::Map {
    let out = js_sys::Map::new();
    for (k, v) in value {
        out.set(&k.into(), &v.into());
    }

    out
}

#[wasm_bindgen(getter_with_clone)]
pub struct Example {
    pub name: Option<String>,
    pub description: Option<String>,
    pub inputs: js_sys::Map,     // HashMap<String, TensorOrMisc>,
    pub sample_out: js_sys::Map, // HashMap<String, TensorOrMisc>,
}

impl From<carton_core::info::Example<GenericStorage>> for Example {
    fn from(value: carton_core::info::Example<GenericStorage>) -> Self {
        Self {
            name: value.name,
            description: value.description,
            inputs: to_map(
                value
                    .inputs
                    .into_iter()
                    .map(|(k, v)| (k, TensorOrMisc::from(v)))
                    .collect(),
            ),
            sample_out: to_map(
                value
                    .sample_out
                    .into_iter()
                    .map(|(k, v)| (k, TensorOrMisc::from(v)))
                    .collect(),
            ),
        }
    }
}

pub enum TensorOrMisc {
    Tensor(PossiblyLoadedWrapper),
    Misc(MiscFileLoaderWrapper),
}

impl From<TensorOrMisc> for JsValue {
    fn from(value: TensorOrMisc) -> Self {
        match value {
            TensorOrMisc::Tensor(v) => v.into(),
            TensorOrMisc::Misc(v) => v.into(),
        }
    }
}

impl From<carton_core::info::TensorOrMisc<GenericStorage>> for TensorOrMisc {
    fn from(value: carton_core::info::TensorOrMisc<GenericStorage>) -> Self {
        match value {
            carton_core::info::TensorOrMisc::Tensor(v) => Self::Tensor(v.into()),
            carton_core::info::TensorOrMisc::Misc(v) => Self::Misc(MiscFileLoaderWrapper(v)),
        }
    }
}

#[wasm_bindgen(getter_with_clone)]
pub struct SelfTest {
    pub name: Option<String>,
    pub description: Option<String>,
    pub inputs: js_sys::Map, // HashMap<String, PossiblyLoadedWrapper>,

    // Can be empty
    pub expected_out: Option<js_sys::Map>, // Option<HashMap<String, PossiblyLoadedWrapper>>,
}

impl From<carton_core::info::SelfTest<GenericStorage>> for SelfTest {
    fn from(value: carton_core::info::SelfTest<GenericStorage>) -> Self {
        Self {
            name: value.name,
            description: value.description,
            inputs: to_map(
                value
                    .inputs
                    .into_iter()
                    .map(|(k, v)| (k, PossiblyLoadedWrapper::from(v)))
                    .collect(),
            ),
            expected_out: value.expected_out.map(|v| {
                to_map(
                    v.into_iter()
                        .map(|(k, v)| (k, PossiblyLoadedWrapper::from(v)))
                        .collect(),
                )
            }),
        }
    }
}

#[derive(Clone)]
#[wasm_bindgen(getter_with_clone)]
pub struct TensorWrapper {
    pub buffer: JsValue,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub stride: Vec<usize>,

    // Keep the tensor alive
    _keepalive: PossiblyLoaded<Tensor<GenericStorage>>,
}

#[wasm_bindgen]
pub struct PossiblyLoadedWrapper(PossiblyLoaded<TensorWrapper>);

impl From<PossiblyLoaded<Tensor<GenericStorage>>> for PossiblyLoadedWrapper {
    fn from(value: PossiblyLoaded<Tensor<GenericStorage>>) -> Self {
        Self(PossiblyLoaded::from_loader(Box::pin(async move {
            let t = value.get().await;

            for_each_numeric_carton_type! {
                return match t {
                    $(
                        carton_core::types::Tensor::$CartonType(item) => {
                            // TODO: handle things not in standard layout
                            // view.as_standard_layout() can create a copy so we need to ensure that stays alive if we use it
                            let view = item.view();
                            let data = view.as_slice().unwrap();

                            // Convert to a u8 slice
                            let u8slice = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * std::mem::size_of::<$RustType>()) };

                            // Avoiding a copy here is hard because views can be invalidated by new memory allocations
                            // that cause the WASM buffer size to change
                            // See https://rustwasm.github.io/wasm-bindgen/api/js_sys/struct.Uint8Array.html#method.view
                            let buffer = js_sys::Uint8Array::from(u8slice);

                            TensorWrapper {
                                buffer: buffer.into(),
                                shape: view.shape().iter().map(|v| *v as _).collect(),
                                dtype: $TypeStr.to_owned(),
                                stride: view.strides().iter().map(|v| *v as _).collect(),

                                _keepalive: value
                            }
                        },
                    )*
                    carton_core::types::Tensor::NestedTensor(_) => panic!("Nested tensor output not implemented yet"),
                    carton_core::types::Tensor::String(item) => {
                        let view = item.view();
                        let data: Vec<_> = view.as_standard_layout().into_iter().collect();

                        TensorWrapper {
                            buffer: serde_wasm_bindgen::to_value(&data).unwrap().into(),
                            shape: view.shape().iter().map(|v| *v as _).collect(),
                            dtype: "string".into(),
                            stride: view.strides().iter().map(|v| *v as _).collect(),

                            // TODO: do we need this keepalive for string tensors?
                            _keepalive: value
                        }
                    }
                }
            }
        })))
    }
}

#[wasm_bindgen]
impl PossiblyLoadedWrapper {
    pub async fn get(&self) -> TensorWrapper {
        self.0.get().await.clone()
    }
}

#[wasm_bindgen]
pub struct MiscFileLoaderWrapper(ArcMiscFileLoader);

#[wasm_bindgen]
impl MiscFileLoaderWrapper {
    pub async fn get(&self) -> wasm_streams::readable::sys::ReadableStream {
        let reader = self.0.get().await;
        ReadableStream::from_async_read(reader.compat(), 1024).into_raw()
    }
}
