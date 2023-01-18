use std::collections::HashMap;

use crate::{
    client::Client,
    do_not_modify::comms::OwnedComms,
    do_not_modify::types::{Device, RPCRequestData, RPCResponseData, SealHandle, Tensor},
};

pub struct Runner {
    client: Client,
}

impl Runner {
    #[cfg(not(target_family = "wasm"))]
    pub async fn new(
        runner_path: &std::path::Path,
        visible_device: Device,
    ) -> Result<Runner, String> {
        use tokio::process::Command;

        // Make sure the runner exists
        if !runner_path.exists() {
            return Err("Runner doesn't exist".into());
        }

        // Create comms
        let (comms, uds_path) = OwnedComms::new().await;

        // Create a command to start the runner
        let mut command = Command::new(runner_path);

        // Check if we have a UUID for a GPU
        if let Device::GPU { uuid: Some(uuid) } = visible_device {
            // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars
            command.env("CUDA_VISIBLE_DEVICES", uuid);
        }

        command
            .args(["--uds-path", uds_path.to_str().unwrap()])
            .spawn()
            .expect("Runner failed to start");

        // Create a client
        let client = Client::new(comms).await;

        Ok(Self { client })
    }

    #[cfg(target_family = "wasm")]
    pub async fn new() -> Result<Runner, String> {
        // Create comms
        let comms = OwnedComms::new().await;

        // Create a client
        let client = Client::new(comms).await;

        Ok(Self { client })
    }

    pub async fn load(&self, _fs: impl lunchbox::ReadableFileSystem) {
        // match client
        //     .do_rpc(RPCRequestData::Load {
        //         path,
        //         runner,
        //         runner_version,
        //         runner_opts,
        //         // TODO change this
        //         visible_device: Device::CPU,
        //     })
        //     .await
        // {
        //     RPCResponseData::Load {
        //         name,
        //         runner,
        //         inputs,
        //         outputs,
        //     } => Ok(Self { client }),
        //     RPCResponseData::Error { e } => Err(e),
        //     _ => panic!("Unexpected RPC response type!"),
        // }
    }

    // pub async fn seal(&self, tensors: HashMap<String, Tensor>) -> Result<SealHandle, String> {
    //     match self.client.do_rpc(RPCRequestData::Seal { tensors }).await {
    //         RPCResponseData::Seal { handle } => Ok(handle),
    //         RPCResponseData::Error { e } => Err(e),
    //         _ => panic!("Unexpected RPC response type!"),
    //     }
    // }

    // pub async fn infer_with_inputs(
    //     &self,
    //     tensors: HashMap<String, Tensor>,
    // ) -> Result<HashMap<String, Tensor>, String> {
    //     match self
    //         .client
    //         .do_rpc(RPCRequestData::InferWithTensors { tensors })
    //         .await
    //     {
    //         RPCResponseData::Infer { tensors } => Ok(tensors),
    //         RPCResponseData::Error { e } => Err(e),
    //         _ => panic!("Unexpected RPC response type!"),
    //     }
    // }

    // pub async fn infer_with_handle(
    //     &self,
    //     handle: SealHandle,
    // ) -> Result<HashMap<String, Tensor>, String> {
    //     match self
    //         .client
    //         .do_rpc(RPCRequestData::InferWithHandle { handle })
    //         .await
    //     {
    //         RPCResponseData::Infer { tensors } => Ok(tensors),
    //         RPCResponseData::Error { e } => Err(e),
    //         _ => panic!("Unexpected RPC response type!"),
    //     }
    // }
}
