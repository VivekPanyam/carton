use std::{collections::HashMap, sync::atomic::AtomicU64};

use carton_runner_interface::server::{init_runner, RequestData, ResponseData, SealHandle};

#[tokio::main]
async fn main() {
    let mut server = init_runner().await;

    let token_gen = AtomicU64::new(0);
    let mut sealed_tensors = HashMap::new();

    while let Some(req) = server.get_next_request().await {
        let req_id = req.id;
        match req.data {
            RequestData::Load { .. } => {
                server
                    .send_response_for_request(req_id, ResponseData::Load)
                    .await
                    .unwrap();
            }

            RequestData::Pack { input_path, .. } => {
                // Just return the input path
                server
                    .send_response_for_request(
                        req_id,
                        ResponseData::Pack {
                            output_path: input_path,
                        },
                    )
                    .await
                    .unwrap();
            }

            RequestData::Seal { tensors } => {
                // Generate a token and store the tensors
                let handle =
                    SealHandle::new(token_gen.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
                sealed_tensors.insert(handle, tensors);
                server
                    .send_response_for_request(req_id, ResponseData::Seal { handle })
                    .await
                    .unwrap();
            }

            RequestData::InferWithTensors { tensors } => {
                // Let's just return the input tensors for now
                server
                    .send_response_for_request(req_id, ResponseData::Infer { tensors })
                    .await
                    .unwrap();
            }

            RequestData::InferWithHandle { handle } => {
                // TODO: return an error instead of using unwrap
                let tensors = sealed_tensors.remove(&handle).unwrap();

                // Let's just return the input tensors for now
                server
                    .send_response_for_request(req_id, ResponseData::Infer { tensors })
                    .await
                    .unwrap();
            }
            _ => {
                // TODO: return an error instead of panicking
                panic!("Got an unknown RPC message type!")
            }
        }
    }
}
