use std::collections::HashMap;
use std::path::Path;

use wasmtime::Engine;

use carton_runner_interface::server::{init_runner, RequestData, ResponseData};
use carton_runner_wasm::{OutputMetadata, WASMModelInstance};

#[tokio::main]
async fn main() {
	let mut server = init_runner().await;

	let engine = Engine::default();
	let mut model: Option<WASMModelInstance> = None;

    while let Some(req) = server.get_next_request().await {
		let req_id = req.id;
		match req.data {
			RequestData::Load { fs, runner_opts, .. } => {
				let fs = server
					.get_readonly_filesystem(fs)
					.await
					.unwrap();
				let bin = &fs.read("model.wasm")
					.await
					.expect("Failed to load model binary");
				let output_md = &fs.read("output_md.json")
					.await
					.expect("Failed to load output metadata");
				let output_md: HashMap<String, OutputMetadata> = serde_json::from_slice(output_md)?;
				model = Some(WASMModelInstance::from_bytes(&engine, bin, output_md)
					.expect("Failed to initialize WASM instance"));
				server
                    .send_response_for_request(req_id, ResponseData::Load)
                    .await
                    .unwrap();
			}
			RequestData::Pack { input_path, temp_folder, fs } => {
				// Same as torch runner
				let fs = server.get_writable_filesystem(fs).await.unwrap();
				fs.symlink(input_path, Path::new(&temp_folder))
                    .await
                    .unwrap();
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
				todo!()
			}
			RequestData::InferWithTensors { tensors, .. } => {
				let result = model
					.as_ref()
					.map(|mut m| m.infer(tensors))
					.unwrap();
                server
                    .send_response_for_request(
                        req_id,
                        ResponseData::Infer {
                            tensors: result.unwrap(),
                        },
                    )
                    .await
                    .unwrap();
			}
			RequestData::InferWithHandle { handle, .. } => {
				todo!()
			}
		}
	}
}