use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use anywhere::types::{AnywhereFS, ReadOnlyFS, ReadWriteFS};
use clap::Parser;
use tokio::sync::mpsc::{self, error::SendError};

use crate::{
    do_not_modify::comms::Comms,
    do_not_modify::types::{ChannelId, FsToken, RPCRequest, RPCResponse},
    multiplexer::Multiplexer,
    types::{Device, Handle, RPCRequestData, RPCResponseData, RpcId, RunnerOpt, Tensor},
};

pub struct Server {
    comms: Comms,
    fs_multiplexer: Multiplexer<
        anywhere::transport::serde::RequestMessageType,
        anywhere::transport::serde::ResponseMessageType,
    >,

    outgoing: mpsc::Sender<RPCResponse>,
    incoming: mpsc::Receiver<RPCRequest>,
}

/// A handle that represents a map of sealed tensors
#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub struct SealHandle(pub(crate) u64);

impl SealHandle {
    pub fn new(v: u64) -> Self {
        SealHandle(v)
    }

    pub fn get(&self) -> u64 {
        self.0
    }
}

impl From<crate::types::SealHandle> for SealHandle {
    fn from(value: crate::types::SealHandle) -> Self {
        Self(value.0)
    }
}

impl From<SealHandle> for crate::types::SealHandle {
    fn from(value: SealHandle) -> Self {
        Self(value.0)
    }
}

/// A request from the core library
#[derive(Debug)]
pub struct Request {
    pub id: RpcId,

    pub data: RequestData,
}

impl Request {
    async fn from(req: RPCRequest, comms: &Comms) -> Self {
        Request {
            id: req.id,
            data: RequestData::from(req.data, comms).await,
        }
    }
}

#[derive(Debug)]
pub enum RequestData {
    Load {
        /// This filesystem points to a folder that is of the same structure as the output of `Pack` (for a particular runner)
        /// For a readonly filesystem
        fs: FsToken,

        /// Load options
        runner_name: String,
        required_framework_version: semver::VersionReq,
        runner_compat_version: u64,
        runner_opts: Option<HashMap<String, RunnerOpt>>,
        visible_device: Device,

        // The hash of the model
        // This should always be avalable unless we're loading an unpacked model
        carton_manifest_hash: Option<String>,
    },

    // Pack a model
    Pack {
        /// A token for a read/write filesystem that the below paths reference
        fs: FsToken,

        // The path to user input data
        // If this is a folder, the runner is allowed to place data in a `.carton` subfolder
        // This can be used if it wants to generate a lockfile for example
        input_path: String,

        // A temporary folder generated by the core library. The runner can use this if it needs
        // to generate output in a new folder.
        // (In some cases, the input can be wrapped as-is and doesn't need to be copied into a new folder)
        // This folder is owned by the core library and will be deleted by it
        temp_folder: String,
    },

    Seal {
        tensors: HashMap<String, Tensor>,
    },

    InferWithTensors {
        tensors: HashMap<String, Tensor>,
    },

    InferWithHandle {
        handle: SealHandle,
    },
}

impl RequestData {
    async fn from(value: RPCRequestData, comms: &Comms) -> Self {
        let from_handles = |tensors: HashMap<String, Handle<Tensor>>| async {
            let mut out = HashMap::new();
            for (k, v) in tensors {
                out.insert(k, v.into_inner(comms).await);
            }

            out
        };

        match value {
            RPCRequestData::Load {
                fs,
                runner_name,
                required_framework_version,
                runner_compat_version,
                runner_opts,
                visible_device,
                carton_manifest_hash,
            } => Self::Load {
                fs,
                runner_name,
                required_framework_version,
                runner_compat_version,
                runner_opts,
                visible_device,
                carton_manifest_hash,
            },
            RPCRequestData::Pack {
                fs,
                input_path,
                temp_folder,
            } => Self::Pack {
                fs,
                input_path,
                temp_folder,
            },
            RPCRequestData::Seal { tensors } => Self::Seal {
                tensors: from_handles(tensors).await,
            },
            RPCRequestData::InferWithTensors { tensors } => Self::InferWithTensors {
                tensors: from_handles(tensors).await,
            },
            RPCRequestData::InferWithHandle { handle } => Self::InferWithHandle {
                handle: handle.into(),
            },
        }
    }
}

#[derive(Debug)]
pub enum ResponseData {
    /// Successful load
    Load,

    Pack {
        // The path to the output directory. This can be in the temp folder passed into `Pack`
        // Note: this must be a *directory* even if the input was a file
        // This references a path on the FS that was passed in
        // during the request
        output_path: String,
    },

    Seal {
        handle: SealHandle,
    },

    Infer {
        tensors: HashMap<String, Tensor>,
    },

    /// Something went wrong
    Error {
        e: String,
    },

    /// This should be used only when something is expected to take a long time (e.g generating a lockfile for a python project)
    SlowLog {
        e: String,
    },
}

impl ResponseData {
    async fn to_rpc(self, comms: &Comms) -> RPCResponseData {
        let into_handles = |tensors: HashMap<String, Tensor>| async {
            let mut out = HashMap::new();
            for (k, v) in tensors {
                out.insert(k, Handle::new(v, comms).await);
            }

            out
        };

        match self {
            ResponseData::Load => RPCResponseData::Load,
            ResponseData::Pack { output_path } => RPCResponseData::Pack { output_path },
            ResponseData::Seal { handle } => RPCResponseData::Seal {
                handle: handle.into(),
            },
            ResponseData::Infer { tensors } => RPCResponseData::Infer {
                tensors: into_handles(tensors).await,
            },
            ResponseData::Error { e } => RPCResponseData::Error { e },
            ResponseData::SlowLog { e } => RPCResponseData::SlowLog { e },
        }
    }
}

impl Server {
    async fn connect(path: &Path) -> Self {
        let comms = Comms::connect(path).await;

        // Set up filesystem handling
        let (tx, rx) = comms.get_channel(ChannelId::FileSystem).await;
        let fs_multiplexer = Multiplexer::new(tx, rx).await;

        let (tx, rx) = comms.get_channel(ChannelId::Rpc).await;

        Server {
            comms,
            fs_multiplexer,
            incoming: rx,
            outgoing: tx,
        }
    }

    pub async fn get_next_request(&mut self) -> Option<Request> {
        match self.incoming.recv().await {
            Some(req) => Some(Request::from(req, &self.comms).await),
            None => None,
        }
    }

    pub async fn send_response_for_request(
        &self,
        req_id: u64,
        res: ResponseData,
    ) -> Result<(), SendError<()>> {
        self.outgoing
            .send(RPCResponse {
                id: req_id,
                data: res.to_rpc(&self.comms).await,
            })
            .await
            .map_err(|_| SendError(()))
    }

    pub async fn get_writable_filesystem(&self, token: FsToken) -> std::io::Result<ReadWriteFS> {
        self.get_filesystem_internal(token).await
    }

    pub async fn get_readonly_filesystem(&self, token: FsToken) -> std::io::Result<ReadOnlyFS> {
        self.get_filesystem_internal(token).await
    }

    async fn get_filesystem_internal<const W: bool, const S: bool>(
        &self,
        token: FsToken,
    ) -> std::io::Result<AnywhereFS<W, S>> {
        let (tx, rx) = self.fs_multiplexer.get_stream_for_id(token.0).await;

        anywhere::transport::serde::connect(tx, rx).await
    }
}

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    uds_path: String,
}

/// Initialize the runner from command line args and return two queues to use to communicate
pub async fn init_runner() -> Server {
    let args = Args::parse();

    // Shutdown the runner if the parent process dies
    // NOTE: this technically shuts down if the thread that forked this process dies, but since
    // the parent should be running in tokio, this should be okay because if the parent's tokio
    // runtime goes down, we should go down.
    #[cfg(not(target_os = "macos"))]
    if unsafe { libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) } != 0 {
        panic!("prctl failed")
    }

    // Watchdog on macos where we can't use PR_SET_PDEATHSIG
    #[cfg(target_os = "macos")]
    std::thread::spawn(|| {
        loop {
            let ppid = unsafe { libc::getppid() };
            if ppid == 1 {
                // The parent exited so we should exit
                std::process::exit(0);
            }

            std::thread::sleep(std::time::Duration::from_secs(1));
        }
    });

    // Initialize logging
    // TODO: pass through slowlog to the main process
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // TODO: run the FD passing channel on top of UDS and get the appropriate channels out
    Server::connect(&PathBuf::from(args.uds_path)).await
}
