use std::sync::{atomic::AtomicU64, Arc};

use anywhere::{transport::serde::SerdeTransport, Servable};
use dashmap::DashMap;
use lunchbox::types::{MaybeSend, MaybeSync};
use tokio::sync::{mpsc, oneshot};

use crate::{
    do_not_modify::comms::OwnedComms,
    do_not_modify::{
        comms::Comms,
        types::{
            ChannelId, FsToken, RPCRequest, RPCRequestData, RPCResponse, RPCResponseData, RpcId,
        },
    },
    do_spawn,
    multiplexer::Multiplexer,
};

pub(crate) struct Client {
    // Comms
    comms: OwnedComms,

    // RPC handling
    inflight: Arc<DashMap<RpcId, oneshot::Sender<RPCResponse>>>,
    rpc_id_gen: AtomicU64,
    rpc_sender: mpsc::Sender<RPCRequest>,

    // Filesystem handling
    fs_multiplexer: Multiplexer<
        anywhere::transport::serde::ResponseMessageType,
        anywhere::transport::serde::RequestMessageType,
    >,
}

impl Client {
    /// Create a new client
    pub(crate) async fn new(comms: OwnedComms) -> Client {
        // Set up RPC handling
        // Get the rpc channel
        let (send, mut recv) = comms
            .get_channel::<RPCRequest, RPCResponse>(ChannelId::Rpc)
            .await;

        // Hold inflight requests
        let inflight: Arc<DashMap<RpcId, oneshot::Sender<RPCResponse>>> = Arc::new(DashMap::new());
        let inflight_clone = inflight.clone();

        // Handle rpc responses
        tokio::spawn(async move {
            while let Some(response) = recv.recv().await {
                // Send the response to the callback
                let callback = inflight_clone.remove(&response.id).unwrap().1;
                callback.send(response).unwrap();
            }
        });

        // Set up filesystem handling
        // We're multiplexing several filesystems over one channel
        // Get the filesystem channel
        let (tx, rx) = comms.get_channel(ChannelId::FileSystem).await;

        // Create the multiplexer
        let mp = Multiplexer::new(tx, rx).await;

        let out = Client {
            comms,
            inflight,
            rpc_id_gen: Default::default(),
            rpc_sender: send,
            fs_multiplexer: mp,
        };

        out
    }

    pub(crate) async fn serve_readonly_fs<T>(&self, fs: Arc<T>) -> FsToken
    where
        T: lunchbox::ReadableFileSystem + MaybeSend + MaybeSync + 'static,
        T::FileType: lunchbox::types::ReadableFile + MaybeSend + MaybeSync + Unpin,
        T::ReadDirPollerType: MaybeSend,
    {
        let (tx, rx, id) = self.fs_multiplexer.get_new_stream().await;

        // Serve the filesystem
        do_spawn(async move {
            fs.build_server()
                .allow_read()
                .disallow_write()
                .disallow_seek()
                .build()
                .into_transport::<SerdeTransport>()
                .serve(tx, rx)
                .await;
        });

        FsToken(id)
    }

    pub(crate) async fn serve_writable_fs<T>(&self, fs: Arc<T>) -> FsToken
    where
        T: lunchbox::WritableFileSystem + MaybeSend + MaybeSync + 'static,
        T::FileType: lunchbox::types::WritableFile + MaybeSend + MaybeSync + Unpin,
        T::ReadDirPollerType: MaybeSend,
    {
        let (tx, rx, id) = self.fs_multiplexer.get_new_stream().await;

        // Serve the filesystem
        do_spawn(async move {
            fs.build_server()
                .allow_read()
                .allow_write()
                .disallow_seek()
                .build()
                .into_transport::<SerdeTransport>()
                .serve(tx, rx)
                .await;
        });

        FsToken(id)
    }

    /// Make an RPC request and get the response
    pub(crate) async fn do_rpc(&self, data: RPCRequestData) -> RPCResponseData {
        // Set the RPC ID
        let id = self
            .rpc_id_gen
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let req = RPCRequest { id, data };

        // Setup our response handler
        let (tx, rx) = oneshot::channel();
        self.inflight.insert(req.id, tx);

        // Send the request
        self.rpc_sender.send(req).await.unwrap();

        // Wait for the response
        match rx.await {
            Ok(v) => v.data,
            Err(_) => panic!("The sender dropped!"),
        }
    }

    pub(crate) fn get_comms(&self) -> &Comms {
        &self.comms
    }
}
