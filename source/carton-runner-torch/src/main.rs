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

use carton_runner_interface::{
    server::{init_runner, RequestData, ResponseData, SealHandle},
    types::{RunnerOpt, Tensor, TensorStorage},
};
use lunchbox::{path::Path, types::WritableFileSystem, ReadableFileSystem};
use std::{collections::HashMap, sync::Arc};

#[tokio::main]
async fn main() {
    let mut server = init_runner().await;

    let mut seal_counter = 0;
    let mut sealed_tensors = HashMap::new();

    let mut model = None;
    let device = tch::Device::cuda_if_available();

    while let Some(req) = server.get_next_request().await {
        let req_id = req.id;
        match req.data {
            RequestData::Load {
                fs, runner_opts, ..
            } => {
                // Handle options
                if let Some(opts) = runner_opts {
                    opts.get("num_threads")
                        .and_then(get_int_opt)
                        .map(|v| tch::set_num_threads(v as _));
                    opts.get("num_interop_threads")
                        .and_then(get_int_opt)
                        .map(|v| tch::set_num_interop_threads(v as _));
                }

                // TODO: error handling
                let fs = server.get_readonly_filesystem(fs).await.unwrap();
                let model_data = fs.read("model.pt").await.unwrap();
                model = tokio::task::spawn_blocking(move || {
                    Some(Arc::new(
                        tch::CModule::load_data_on_device(&mut model_data.as_slice(), device)
                            .unwrap(),
                    ))
                })
                .await
                .unwrap();

                server
                    .send_response_for_request(req_id, ResponseData::Load)
                    .await
                    .unwrap();
            }

            RequestData::Pack {
                input_path,
                temp_folder,
                fs,
            } => {
                let fs = server.get_writable_filesystem(fs).await.unwrap();

                // The input path should be a pt file. Let's symlink it into the temp dir
                // (this symlink will be resolved when packing the model since it's an external one)
                fs.symlink(input_path, Path::new(&temp_folder).join("model.pt"))
                    .await
                    .unwrap();

                // Return the temp folder
                server
                    .send_response_for_request(
                        req_id,
                        ResponseData::Pack {
                            output_path: temp_folder,
                        },
                    )
                    .await
                    .unwrap();
            }

            RequestData::Seal { tensors } => {
                // Generate a token and store the tensors
                sealed_tensors.insert(seal_counter, tensors);
                server
                    .send_response_for_request(
                        req_id,
                        ResponseData::Seal {
                            handle: SealHandle::new(seal_counter),
                        },
                    )
                    .await
                    .unwrap();

                seal_counter += 1
            }

            RequestData::InferWithTensors { tensors } => {
                // TODO: error handling
                let m = model.as_ref().unwrap().clone();
                let out = tokio::task::spawn_blocking(move || infer(m, tensors, device))
                    .await
                    .unwrap();

                server
                    .send_response_for_request(req_id, ResponseData::Infer { tensors: out })
                    .await
                    .unwrap();
            }

            RequestData::InferWithHandle { handle } => {
                // TODO: error handling
                let tensors = sealed_tensors.remove(&handle.get()).unwrap();
                let m = model.as_ref().unwrap().clone();
                let out = tokio::task::spawn_blocking(move || infer(m, tensors, device))
                    .await
                    .unwrap();

                // Let's just return the input tensors for now
                server
                    .send_response_for_request(req_id, ResponseData::Infer { tensors: out })
                    .await
                    .unwrap();
            }
        }
    }
}

fn infer(
    model: Arc<tch::CModule>,
    tensors: HashMap<String, Tensor>,
    device: tch::Device,
) -> HashMap<String, Tensor> {
    let tensors = tensors_to_tch(tensors, device);

    // TODO: error handling
    let out = model.forward_is(&[tensors]).unwrap();

    // Type conversion on the way out
    let out: Vec<(tch::IValue, tch::IValue)> = out.try_into().unwrap();
    out.into_iter()
        .map(|(k, v)| {
            (
                k.try_into().unwrap(),
                tensor_from_ivalue(v.try_into().unwrap()),
            )
        })
        .collect()
}

fn tensors_to_tch(tensors: HashMap<String, Tensor>, device: tch::Device) -> tch::IValue {
    tensors
        .into_iter()
        .map(|(k, v)| (k.into(), tensor_to_ivalue(v, device)))
        .collect::<Vec<(tch::IValue, tch::IValue)>>()
        .into()
}

// Conversion from carton tensors to torch IValues.
fn tensor_to_ivalue(value: Tensor, device: tch::Device) -> tch::IValue {
    match value {
        Tensor::Float(v) => storage_to_tensor(v, tch::Kind::Float, device),
        Tensor::Double(v) => storage_to_tensor(v, tch::Kind::Double, device),
        Tensor::I8(v) => storage_to_tensor(v, tch::Kind::Int8, device),
        Tensor::I16(v) => storage_to_tensor(v, tch::Kind::Int16, device),
        Tensor::I32(v) => storage_to_tensor(v, tch::Kind::Int, device),
        Tensor::I64(v) => storage_to_tensor(v, tch::Kind::Int64, device),
        Tensor::U8(v) => storage_to_tensor(v, tch::Kind::Uint8, device),

        // TODO: don't panic
        Tensor::U16(_) => panic!("Torch doesn't support uint16"),
        Tensor::U32(_) => panic!("Torch doesn't support uint32"),
        Tensor::U64(_) => panic!("Torch doesn't support uint64"),
        Tensor::NestedTensor(_) => panic!("Nested tensors are not yet supported"),

        Tensor::String(v) => {
            // Special handling for strings
            let view = v.view();

            // Currently only support flat lists or scalars
            match view.ndim() {
                0 => {
                    // Scalar
                    view.first().unwrap().to_owned().into()
                },
                1 => {
                    view.as_slice().unwrap().to_vec().into()
                }
                dim => panic!("Tried using a string tensor with {dim} dims. Currently, only string tensors of 0 or 1 dims are supported.")
            }
        }
    }
}

fn storage_to_tensor<T>(v: TensorStorage<T>, kind: tch::Kind, device: tch::Device) -> tch::IValue {
    let view = v.view();
    let ptr = view.as_ptr();
    let size: Vec<_> = view.shape().into_iter().map(|v| (*v) as _).collect();
    let strides: Vec<_> = view.strides().into_iter().map(|v| (*v) as _).collect();

    // Note the `copy: true`. This decouples the torch tensor from our input so we can safely drop the input data.
    // tch doesn't currently support deleters in `from_blob` so we don't have a better alternative
    // This also attempts to do a non blocking copy to the target device
    let t = unsafe {
        tch::Tensor::from_blob(ptr as _, &size, &strides, kind, tch::Device::Cpu)
            .to_device_(device, kind, true, true)
    };

    t.into()
}

// Macro for conversions from torch to carton
macro_rules! impl_output_copy {
    ($tensor:ident, $type:ty) => {{
        // Create an output tensor with the same shape
        let mut output_tensor =
            TensorStorage::<$type>::new($tensor.size().iter().map(|v| (*v) as _).collect());
        let mut output_view = output_tensor.view_mut();
        let sliced_output_view = output_view.as_slice_mut().unwrap();

        // Copy the data in
        $tensor
            .to(tch::Device::Cpu)
            .f_copy_data(sliced_output_view, $tensor.numel())
            .unwrap();

        output_tensor.into()
    }};
}

fn tensor_from_ivalue(value: tch::IValue) -> Tensor {
    match value {
        tch::IValue::Tensor(tensor) => match tensor.kind() {
            tch::Kind::Uint8 => impl_output_copy!(tensor, u8),
            tch::Kind::Int8 => impl_output_copy!(tensor, i8),
            tch::Kind::Int16 => impl_output_copy!(tensor, i16),
            tch::Kind::Int => impl_output_copy!(tensor, i32),
            tch::Kind::Int64 => impl_output_copy!(tensor, i64),
            tch::Kind::Float => impl_output_copy!(tensor, f32),
            tch::Kind::Double => impl_output_copy!(tensor, f64),
            other => panic!("Tensor kind {other:?} is currently unsupported as an output!"),
        },
        tch::IValue::String(scalar_string) => {
            let mut output_tensor = TensorStorage::new(vec![]);
            let mut view = output_tensor.view_mut();
            *view.first_mut().unwrap() = scalar_string;

            output_tensor.into()
        }
        tch::IValue::StringList(string_list) => {
            let mut output_tensor = TensorStorage::new(vec![string_list.len() as _]);
            let mut view = output_tensor.view_mut();
            view.as_slice_mut().unwrap().clone_from_slice(&string_list);

            output_tensor.into()
        }
        tch::IValue::GenericList(list) => {
            let mut output_tensor = TensorStorage::new(vec![list.len() as _]);
            let mut view = output_tensor.view_mut();

            for (a, item) in std::iter::zip(view.as_slice_mut().unwrap(), list) {
                // We want to make sure each value in this list is a string
                if let tch::IValue::String(s) = item {
                    *a = s;
                } else {
                    panic!("Got a GenericList that wasn't entirely strings");
                }
            }

            output_tensor.into()
        }
        other => panic!("Unsupported IValue type {other:?}"),
    }
}

fn get_int_opt(opt: &RunnerOpt) -> Option<i64> {
    match opt {
        RunnerOpt::Integer(v) => Some(*v),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_scalar() {
        let data = vec![1.4];
        let shape = vec![];

        let arr = ndarray::ArrayD::from_shape_vec(shape, data).unwrap();

        assert_eq!(arr.ndim(), 0);
        assert_eq!(arr.len(), 1);
    }

    #[test]
    fn test_assign_scalar() {
        let mut arr = ndarray::ArrayD::<f64>::zeros(vec![]);
        assert_eq!(arr.ndim(), 0);
        assert_eq!(arr.len(), 1);
        *arr.first_mut().unwrap() = 32.0;
    }
}
