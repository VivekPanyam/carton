use carton_runner_interface::server::{init_runner, RequestData, ResponseData};

use packager::update_or_generate_lockfile;

mod env;
mod packager;
mod pip_utils;
mod python_utils;
mod wheel;

#[tokio::main]
async fn main() {
    let mut server = init_runner().await;

    while let Some(req) = server.get_next_request().await {
        let req_id = req.id;
        match req.data {
            RequestData::Load { .. } => {
                // TODO: implement
                server
                    .send_response_for_request(req_id, ResponseData::Load)
                    .await
                    .unwrap();
            }
            RequestData::Pack { fs, input_path, .. } => {
                let fs = server.get_writable_filesystem(fs).await.unwrap();

                // Update or generate a lockfile in the input dir
                update_or_generate_lockfile(&fs, &input_path).await;

                // The dir that carton should pack is just the input path
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
            _ => todo!(),
        }
    }
}
