use std::{collections::HashMap, sync::{atomic::AtomicU64, Arc}, path::PathBuf, process::Command};

use dashmap::DashMap;
use tokio::{sync::{oneshot, mpsc}, net::UnixListener, io::{BufWriter, AsyncWriteExt, BufReader, AsyncReadExt}};

use crate::types::{RPCRequest, RPCResponse, RpcId, SealHandle, Tensor, RPCRequestData, RPCResponseData, Device, TensorSpec};

struct Client {
    inflight: Arc<DashMap<RpcId, oneshot::Sender<RPCResponse>>>,
    rpc_id_gen: AtomicU64,

    sender: mpsc::Sender<RPCRequest>,

    uds_path: String,
}

impl Client {
    /// Create a new client and return a path to the UDS it's using
    async fn new() -> Client {

        // Create a UDS in a temp dir
        let tempdir = tempfile::tempdir().unwrap();
        let bind_path = tempdir.path().join("hand");
        let uds_path = bind_path.to_str().unwrap().to_string();
        let listener = UnixListener::bind(bind_path).unwrap();

        let (tx, mut rx) = mpsc::channel::<RPCRequest>(32);

        let inflight: Arc<DashMap<RpcId, oneshot::Sender<RPCResponse>>> = Arc::new(DashMap::new());
        let inflight_clone = inflight.clone();

        tokio::spawn(async move {
            // Move tempdir so it doesn't go out of scope before something connects
            let _tempdir = tempdir;

            // Wait for a connection
            let stream = match listener.accept().await {
                Ok((stream, _)) => stream,
                Err(e) => panic!("Error when connecting: {}", e),
            };

            // Split into read and write
            let (read_stream, write_stream) = stream.into_split();
            let mut bw = BufWriter::new(write_stream);

            // Spawn a task to handle reads
            tokio::spawn(async move {
                let mut br = BufReader::new(read_stream);

                loop {
                    // Read the size and then read the data
                    let size = br.read_u64().await.unwrap() as usize;
                    let mut vec = Vec::with_capacity(size);
                    vec.resize(size, 0);
                    br.read_exact(&mut vec).await.unwrap();

                    // TODO: offload this to a compute thread if it's too slow
                    let response: RPCResponse = bincode::deserialize(&vec).unwrap();

                    // Send the response to the callback
                    let callback = inflight_clone.remove(&response.id).unwrap().1;
                    callback.send(response).unwrap();
                }
            });

            // Handle writes
            loop {
                let item = match rx.try_recv() {
                    Ok(item) => item,
                    Err(mpsc::error::TryRecvError::Empty) => {
                        // Nothing to recv
                        // Flush the writer
                        bw.flush().await.unwrap();

                        // Blocking wait for new things to send
                        rx.recv().await.unwrap()
                    },
                    Err(mpsc::error::TryRecvError::Disconnected) => {
                        // We're done
                        break;
                    },
                };

                // Serialize and write size + data to the buffer
                // TODO: offload this to a compute thread if it's too slow
                let data = bincode::serialize(&item).unwrap();
                bw.write_u64(data.len() as _).await.unwrap();
                bw.write_all(&data).await.unwrap();
            }

        });

        Client {
            inflight,
            rpc_id_gen: Default::default(),
            sender: tx,
            uds_path,
        }
    }

    fn uds_path(&self) -> &str {
        &self.uds_path
    } 

    /// Make an RPC request and get the response
    async fn do_rpc(&self, data: RPCRequestData) -> RPCResponseData {
        // Set the RPC ID
        let id = self.rpc_id_gen.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let req = RPCRequest { id, data };

        // Setup our response handler
        let (tx, rx) = oneshot::channel();
        self.inflight.insert(req.id, tx);

        // Send the request
        self.sender.send(req).await.unwrap();

        // Wait for the response
        match rx.await {
            Ok(v) => v.data,
            Err(_) => panic!("The sender dropped!"),
        }
    }
}


fn launch_runner(runner_name: &str, uds_path: &str) {
    let runner_base_dir = std::env::var("CARTON_RUNNER_DIR").unwrap_or("/usr/local/carton_runners".to_string());

    // Ex /usr/local/carton_runners/torchscript/runner
    let runner_path: PathBuf = [&runner_base_dir, runner_name, "runner"].iter().collect();

    if !runner_path.as_path().exists() {
        panic!("Runner didn't exist at expected path: {:#?}", runner_path);
    }

    // Start the runner
    Command::new(runner_path)
        .args(["--uds-path", uds_path])
        .spawn()
        .expect("Runner failed to start");
}



pub struct Carton {
    client: Client,

    // Model info
    pub model_name: String,
    pub model_runner: String,
    pub inputs: Vec<TensorSpec>,
    pub outputs: Vec<TensorSpec>,
}

impl Carton {
    pub async fn new(path: String, runner: Option<String>, runner_version: Option<String>, runner_opts: Option<String>, _visible_device: String) -> Result<Carton, String> {
        // Create a client
        let client = Client::new().await;

        // Launch a runner with the UDS path
        launch_runner(&runner.as_ref().unwrap(), client.uds_path());

        match client.do_rpc(RPCRequestData::Load {
            path,
            runner,
            runner_version,
            runner_opts,
            // TODO change this
            visible_device: Device::CPU
        }).await {
            RPCResponseData::Load {
                name,
                runner,
                inputs,
                outputs } => Ok(Carton {
                    client,
                    model_name: name,
                    model_runner: runner,
                    inputs,
                    outputs,
                }),
            RPCResponseData::Error { e } => Err(e),
            _ => panic!("Unexpected RPC response type!")
        }
    }

    pub async fn seal(&self, tensors: HashMap<String, Tensor>) -> Result<SealHandle, String> {
        match self.client.do_rpc(RPCRequestData::Seal { tensors }).await {
            RPCResponseData::Seal { handle } => Ok(handle),
            RPCResponseData::Error { e } => Err(e),
            _ => panic!("Unexpected RPC response type!")
        }
    }

    pub async fn infer_with_inputs(&self, tensors: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>, String> {
        match self.client.do_rpc(RPCRequestData::InferWithTensors { tensors }).await {
            RPCResponseData::Infer { tensors } => Ok(tensors),
            RPCResponseData::Error { e } => Err(e),
            _ => panic!("Unexpected RPC response type!")
        }
    }

    pub async fn infer_with_handle(&self, handle: SealHandle) -> Result<HashMap<String, Tensor>, String> {
        match self.client.do_rpc(RPCRequestData::InferWithHandle { handle }).await {
            RPCResponseData::Infer { tensors } => Ok(tensors),
            RPCResponseData::Error { e } => Err(e),
            _ => panic!("Unexpected RPC response type!")
        }
    }

}