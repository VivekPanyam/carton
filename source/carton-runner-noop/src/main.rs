use carton::{server::init_runner, types::{RPCRequestData, RPCResponse, RPCResponseData}};

#[tokio::main]
async fn main() {
    let (mut incoming, outgoing) = init_runner().await;

    while let Some(item) = incoming.recv().await {
        println!("Got item: {:#?}", &item);

        match item.data {
            RPCRequestData::Load {
                path,
                runner,
                runner_version,
                runner_opts,
                visible_device
            } => {

                // TODO: actually load a model

                outgoing.send(RPCResponse {
                    id: item.id,
                    data: RPCResponseData::Load { name: "model_name".to_string(), runner: "torchscript".to_string(), inputs: Vec::new(), outputs: Vec::new() }
                }).await.unwrap();
            },
            RPCRequestData::Seal { tensors } => {

            },
            RPCRequestData::InferWithTensors { tensors } => {
                // Let's just return the input tensors for now
                outgoing.send(RPCResponse {
                    id: item.id,
                    data: RPCResponseData::Infer { tensors }
                }).await.unwrap();
            },
            RPCRequestData::InferWithHandle { handle } => {

            },
        }

    }
}