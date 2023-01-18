use std::path::{Path, PathBuf};

use anywhere::types::{AnywhereFS, ReadOnlyFS, ReadWriteFS};
use clap::Parser;
use tokio::sync::mpsc::{self, error::SendError};

use crate::{
    do_not_modify::comms::Comms,
    do_not_modify::types::{FsToken, RPCRequest, RPCResponse, ChannelId},
    multiplexer::Multiplexer, types::{RPCRequestData, RPCResponseData},
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

    pub async fn get_next_request(&mut self) -> Option<RPCRequest> {
        self.incoming.recv().await
    }

    pub async fn send_response_for_request(&self, req_id: u64, res: RPCResponseData) -> Result<(), SendError<RPCResponseData>> {
        self.outgoing.send(RPCResponse { id: req_id, data: res }).await.map_err(|e| SendError(e.0.data))
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

    // TODO: run the FD passing channel on top of UDS and get the appropriate channels out
    Server::connect(&PathBuf::from(args.uds_path)).await
}
