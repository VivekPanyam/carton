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

//! Serialization and deserialization of tensors based on v1 of the carton format spec

use std::{collections::HashMap, sync::Arc};

use carton_macros::for_each_numeric_carton_type;
use lunchbox::{
    types::{MaybeSend, MaybeSync, ReadableFile},
    ReadableFileSystem,
};
use serde::{Deserialize, Serialize};

use crate::{info::PossiblyLoaded, types::Tensor};

#[derive(Default, Serialize, Deserialize)]
struct IndexToml {
    tensor: Vec<TensorInfo>,
}

/// An individual string tensor
#[derive(Default, Serialize, Deserialize)]
struct TensorInfo {
    name: String,
    dtype: String,

    /// For non-nested tensors
    shape: Option<Vec<u64>>,
    file: Option<String>,

    /// For nested tensors
    inner: Vec<String>,
}

/// The data for a string tensor
#[derive(Default, Serialize, Deserialize)]
struct StringsToml {
    data: Vec<String>,
}

pub(crate) fn save_tensors(
    tensor_data_path: &std::path::Path,
    tensors: HashMap<String, &Tensor>,
) -> crate::error::Result<()> {
    let mut index_toml = IndexToml::default();

    // First, split out all nested tensors
    let (nested, mut unnested) = tensors
        .into_iter()
        .partition::<HashMap<_, _>, _>(|(_, v)| matches!(v, Tensor::NestedTensor(_)));

    // Add all the inner tensors of the nested tensors into unnested
    for (k, v) in nested {
        if let Tensor::NestedTensor(items) = v {
            // Create a nested tensor to write out
            let mut nt = TensorInfo {
                name: k.strip_prefix("@tensor_data/").unwrap().to_owned(),
                dtype: "nested".into(),
                ..Default::default()
            };

            // Loop through
            for (idx, t) in items.into_iter().enumerate() {
                let inner_name = format!("_carton_nested_inner_{k}_{idx}");

                // Confirm that the inner tensor is not a nested tensor
                if matches!(t, Tensor::NestedTensor(_)) {
                    panic!("NestedTensors cannot contain NestedTensors");
                }

                // Store the inner tensor name
                nt.inner.push(inner_name.clone());

                // Add the inner tensor to our map of unnested tensors to serialize
                if unnested.insert(inner_name, t).is_some() {
                    panic!("Tensor names starting with `_carton_nested_inner_` are reserved.")
                }
            }

            // Add it to our toml file
            index_toml.tensor.push(nt);
        } else {
            unreachable!("This shouldn't happen because we partitioned above")
        }
    }

    // Serialize all the inner tensors
    for (tensor_idx, (k, v)) in unnested.iter().enumerate() {
        if let Tensor::String(t) = v {
            // String tensor
            let string_tensor = StringsToml {
                // TODO: this can make a copy
                data: t.view().as_standard_layout().into_iter().collect(),
            };

            let fname = format!("tensor_{tensor_idx}.toml");

            // Add it to the index
            index_toml.tensor.push(TensorInfo {
                name: k.strip_prefix("@tensor_data/").unwrap().to_owned(),
                dtype: "string".into(),
                shape: Some(t.view().shape().into_iter().map(|v| *v as u64).collect()),
                file: Some(fname.clone()),
                ..Default::default()
            });

            // Write out the data
            let serialized = toml::to_string_pretty(&string_tensor).unwrap();
            std::fs::write(tensor_data_path.join(fname), serialized).unwrap();
        } else {
            // Numeric tensor
            for_each_numeric_carton_type! {
                match v {
                    Tensor::NestedTensor(_) => {
                        unreachable!("This shouldn't happen because we partitioned above")
                    }
                    Tensor::String(_) => unreachable!(
                        "This shouldn't happen because we handled string tensors immediately above"
                    ),
                    $(
                        Tensor::$CartonType(v) => {
                            // TODO: this can make a copy
                            let view = v.view();
                            let array = view.as_standard_layout();

                            #[cfg(not(target_endian = "little"))]
                            compile_error!("Writing tensor_data to disk is currently only supported on little-endian platforms");

                            let bytes_per_elem = bytes_per_elem(&view);
                            let total_bytes = array.len() * bytes_per_elem;

                            let data = unsafe { std::slice::from_raw_parts(array.as_ptr() as *const u8, total_bytes) };

                            let fname = format!("tensor_{tensor_idx}.bin");

                            // Add it to the index
                            index_toml.tensor.push(TensorInfo {
                                name: k.strip_prefix("@tensor_data/").unwrap().to_owned(),
                                dtype: $TypeStr.into(),
                                shape: Some(array.shape().into_iter().map(|v| *v as u64).collect()),
                                file: Some(fname.clone()),
                                ..Default::default()
                            });

                            // Write the file out
                            std::fs::write(tensor_data_path.join(fname), data).unwrap();
                        }
                    )*
                };
            }
        }
    }

    // Write the index
    let serialized = toml::to_string_pretty(&index_toml).unwrap();
    std::fs::write(tensor_data_path.join("index.toml"), serialized).unwrap();

    Ok(())
}

fn bytes_per_elem<T>(_array: &ndarray::ArrayViewD<T>) -> usize {
    std::mem::size_of::<T>()
}

/// Loads tensors
pub(crate) async fn load_tensors<T>(
    fs: &Arc<T>,
    tensor_data_path: &lunchbox::path::Path,
) -> crate::error::Result<HashMap<String, PossiblyLoaded<Tensor>>>
where
    T: ReadableFileSystem + MaybeSend + MaybeSync + 'static,
    T::FileType: ReadableFile + MaybeSend + MaybeSync + 'static,
{
    // First, read the index from disk
    let index_toml: IndexToml =
        toml::from_slice(&fs.read(tensor_data_path.join("index.toml")).await.unwrap()).unwrap();

    // Create loaders for all the unnested tensors
    let mut unnested: HashMap<String, PossiblyLoaded<Tensor>> = HashMap::new();
    for t in &index_toml.tensor {
        for_each_numeric_carton_type! {
            let loader = match t.dtype.as_str() {
                "nested" => {
                    // Skip
                    continue;
                },
                "string" => {
                    let shape: Vec<_> = t.shape.as_ref().unwrap().iter().map(|v| *v as usize).collect();
                    let fname = t.file.clone().unwrap();
                    let fs = fs.clone();
                    let path = tensor_data_path.join(fname);
                    PossiblyLoaded::from_loader(Box::pin(async move {
                        let data = fs.read(path).await.unwrap();
                        let strings: StringsToml = toml::from_slice(&data).unwrap();
                        Tensor::String(ndarray::ArrayD::<String>::from_shape_vec(shape, strings.data).unwrap().into())
                    }))
                },
                $(
                    $TypeStr => {
                        let shape: Vec<_> = t.shape.as_ref().unwrap().iter().map(|v| *v as usize).collect();
                        let fname = t.file.clone().unwrap();
                        let fs = fs.clone();
                        let path = tensor_data_path.join(fname);
                        PossiblyLoaded::from_loader(Box::pin(async move {
                            let data = fs.read(path).await.unwrap();

                            #[cfg(not(target_endian = "little"))]
                            compile_error!("Reading tensor_data from disk is currently only supported on little-endian platforms");

                            let bytes_per_elem = std::mem::size_of::<$RustType>();
                            let numel = data.len() / bytes_per_elem;

                            let typed_data = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const $RustType, numel) }.to_vec();

                            Tensor::$CartonType(ndarray::ArrayD::<$RustType>::from_shape_vec(shape, typed_data).unwrap().into())
                        }))
                    },
                )*
                dtype => panic!("Found tensor with unknown type {dtype}. You may need to upgrade the version of Carton you're using.")
            };

            unnested.insert(t.name.clone(), loader);
        }
    }

    // Create loaders for all the nested tensors
    let mut out: HashMap<_, _> = index_toml
        .tensor
        .into_iter()
        .filter_map(|item| {
            if item.dtype == "nested" {
                let inner: Vec<_> = item
                    .inner
                    .into_iter()
                    .map(|name| unnested.remove(&name).unwrap())
                    .collect();
                Some((
                    item.name,
                    PossiblyLoaded::from_loader(Box::pin(async move {
                        // Actually load the inner tensors
                        let mut tensors = Vec::new();
                        for item in inner {
                            tensors.push(item.into_get().await.unwrap());
                        }

                        // Return a nested tensor
                        Tensor::NestedTensor(tensors)
                    })),
                ))
            } else {
                None
            }
        })
        .collect();

    // Merge in the remaining unnested tensors
    out.extend(unnested);

    Ok(out)
}
