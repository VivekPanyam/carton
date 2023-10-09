use color_eyre::eyre::{eyre, Result};
use lunchbox::{path::Path, types::WritableFileSystem, ReadableFileSystem};
use wasmtime::{Config, Engine};

use carton_runner_interface::server::{init_runner, RequestData, ResponseData};
use carton_runner_wasm::WASMModelInstance;

fn new_engine() -> Result<Engine> {
    let mut config = Config::new();
    config.wasm_component_model(true);
    Engine::new(&config).map_err(|e| eyre!(e))
}

#[tokio::main]
async fn main() {
    color_eyre::install().unwrap();
    let mut server = init_runner().await;
    let engine = new_engine().unwrap();
    let mut model: Option<WASMModelInstance> = None;

    while let Some(req) = server.get_next_request().await {
        let req_id = req.id;
        match req.data {
            RequestData::Load { fs, .. } => {
                let fs = server.get_readonly_filesystem(fs).await.unwrap();
                let bin = &fs.read("model.wasm").await.unwrap();
                model = Some(
                    WASMModelInstance::from_bytes(&engine, bin)
                        .expect("Failed to initialize WASM model"),
                );
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
                fs.symlink(input_path, Path::new(&temp_folder).join("model.wasm"))
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
            RequestData::Seal { .. } => {
                todo!()
            }
            RequestData::InferWithTensors { tensors, .. } => {
                let result = model.as_mut().map(|m| m.infer(tensors)).unwrap();
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
            RequestData::InferWithHandle { .. } => {
                todo!()
            }
        }
    }
}
