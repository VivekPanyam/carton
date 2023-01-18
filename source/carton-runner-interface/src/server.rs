use std::path::{Path, PathBuf};

use anywhere::types::{AnywhereFS, ReadOnlyFS, ReadWriteFS};
use clap::Parser;
use tokio::sync::mpsc;

use crate::{
    do_not_modify::comms::Comms,
    do_not_modify::types::{FsToken, RPCRequest, RPCResponse, ChannelId},
    multiplexer::Multiplexer,
};

pub struct Server {
    comms: Comms,
    fs_multiplexer: Multiplexer<
        anywhere::transport::serde::RequestMessageType,
        anywhere::transport::serde::ResponseMessageType,
    >,
}

impl Server {
    async fn connect(path: &Path) -> Self {
        let comms = Comms::connect(path).await;

        // Set up filesystem handling
        let (tx, rx) = comms.get_channel(ChannelId::FileSystem).await;
        let fs_multiplexer = Multiplexer::new(tx, rx).await;

        Server {
            comms,
            fs_multiplexer,
        }
    }

    /// This can only be called ONCE
    pub async fn get_queues(&self) -> (mpsc::Sender<RPCResponse>, mpsc::Receiver<RPCRequest>) {
        // Set up RPC handling
        // Get the rpc channel
        let (tx, rx) = self.comms.get_channel(ChannelId::Rpc).await;

        (tx, rx)
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
