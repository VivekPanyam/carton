use std::{collections::HashMap, sync::atomic::AtomicU64};

use carton_runner_interface::{
    server::init_runner,
    types::{RPCRequestData, RPCResponseData, SealHandle},
};

#[tokio::main]
async fn main() {
    let mut server = init_runner().await;

    let token_gen = AtomicU64::new(0);
    let mut sealed_tensors = HashMap::new();

    while let Some(req) = server.get_next_request().await {
        let req_id = req.id;
        match req.data {
            RPCRequestData::Load { .. } => {
                server
                    .send_response_for_request(req_id, RPCResponseData::Load)
                    .await
                    .unwrap();
            }

            RPCRequestData::Pack { input_path, .. } => {
                // Just return the input path
                server
                    .send_response_for_request(
                        req_id,
                        RPCResponseData::Pack {
                            output_path: input_path,
                        },
                    )
                    .await
                    .unwrap();
            }

            RPCRequestData::Seal { tensors } => {
                // Generate a token and store the tensors
                let handle =
                    SealHandle::new(token_gen.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
                sealed_tensors.insert(handle, tensors);
                server
                    .send_response_for_request(req_id, RPCResponseData::Seal { handle })
                    .await
                    .unwrap();
            }

            RPCRequestData::InferWithTensors { tensors } => {
                // Let's just return the input tensors for now
                server
                    .send_response_for_request(req_id, RPCResponseData::Infer { tensors })
                    .await
                    .unwrap();
            }

            RPCRequestData::InferWithHandle { handle } => {
                // TODO: return an error instead of using unwrap
                let tensors = sealed_tensors.remove(&handle).unwrap();

                // Let's just return the input tensors for now
                server
                    .send_response_for_request(req_id, RPCResponseData::Infer { tensors })
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
