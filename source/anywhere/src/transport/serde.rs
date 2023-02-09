use std::sync::{atomic::AtomicUsize, Arc};

use crate::{
    rpc::{self, AnywhereRPCClient, MaybeRead, MaybeSeek, MaybeWrite},
    types::AnywhereFS,
};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, oneshot};

use super::Transport;

#[derive(Default)]
pub struct SerdeTransportClient {
    id_counter: AtomicUsize,
    callbacks: DashMap<usize, oneshot::Sender<rpc::AnywhereRPCResponse>>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RequestMessageType {
    rpc_id: usize,
    msg: rpc::AnywhereRPCRequest,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ResponseMessageType {
    rpc_id: usize,
    msg: rpc::AnywhereRPCResponse,
}

impl SerdeTransportClient {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn transform_req(&self, msg: rpc::MessageType) -> RequestMessageType {
        let (msg, callback) = msg;

        // Generate an RPC ID
        let rpc_id = self
            .id_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Store the callback
        self.callbacks.insert(rpc_id, callback);

        // Return the transformed message
        RequestMessageType { rpc_id, msg }
    }

    pub fn on_res(&self, res: ResponseMessageType) {
        if let Some((_, callback)) = self.callbacks.remove(&res.rpc_id) {
            let _ = callback.send(res.msg);
        }
    }
}

pub async fn connect<const WRITABLE: bool, const SEEKABLE: bool>(
    send: mpsc::Sender<RequestMessageType>,
    mut recv: mpsc::Receiver<ResponseMessageType>,
) -> std::io::Result<AnywhereFS<WRITABLE, SEEKABLE>> {
    // Create a queue
    let (tx, mut rx) = mpsc::channel(32);

    // Handle the queue with a serde transport
    let transport = Arc::new(SerdeTransportClient::new());
    let recv_transport = transport.clone();
    tokio::spawn(async move {
        while let Some(item) = rx.recv().await {
            let msg = transport.transform_req(item);

            send.send(msg).await.unwrap();
        }
    });

    // Handle responses
    tokio::spawn(async move {
        while let Some(item) = recv.recv().await {
            recv_transport.on_res(item);
        }
    });

    // Create a client
    AnywhereRPCClient::new(tx).try_to_fs().await
}

pub struct SerdeTransportServer<T, A, B, C> {
    inner: rpc::AnywhereRPCServer<T, A, B, C>,
}

impl<T, A: MaybeRead<T>, B: MaybeWrite<T>, C: MaybeSeek<T>> SerdeTransportServer<T, A, B, C> {
    pub fn new(inner: rpc::AnywhereRPCServer<T, A, B, C>) -> Self {
        Self { inner }
    }

    pub async fn handle_request(&self, msg: RequestMessageType) -> ResponseMessageType {
        let res = self.inner.handle_message(msg.msg).await;

        ResponseMessageType {
            rpc_id: msg.rpc_id,
            msg: res,
        }
    }

    pub async fn serve(
        self,
        send: mpsc::Sender<ResponseMessageType>,
        mut recv: mpsc::Receiver<RequestMessageType>,
    ) {
        while let Some(req) = recv.recv().await {
            // Handle the request and get a response
            let res = self.handle_request(req).await;

            // Write the response out
            send.send(res).await.unwrap();
        }
    }
}

/// Serves a filesystem
pub async fn serve_fs<T, A: MaybeRead<T>, B: MaybeWrite<T>, C: MaybeSeek<T>>(
    fs: crate::rpc::AnywhereRPCServer<T, A, B, C>,
    send: mpsc::Sender<ResponseMessageType>,
    recv: mpsc::Receiver<RequestMessageType>,
) {
    SerdeTransportServer::new(fs).serve(send, recv).await
}

pub struct SerdeTransport {}
impl Transport for SerdeTransport {
    type Ret<T, A, B, C> = SerdeTransportServer<T, A, B, C>;

    fn new<T, A, B, C>(inner: rpc::AnywhereRPCServer<T, A, B, C>) -> Self::Ret<T, A, B, C> {
        SerdeTransportServer { inner }
    }
}

#[cfg(test)]
mod tests {
    use super::{RequestMessageType, ResponseMessageType};

    #[test]
    fn test_bincode() {
        // Request
        let ser = bincode::serialize(&RequestMessageType {
            rpc_id: 0,
            msg: crate::rpc::AnywhereRPCRequest::Canonicalize { path: "".into() },
        })
        .unwrap();

        let _: RequestMessageType = bincode::deserialize(&ser).unwrap();

        // Response
        let ser = bincode::serialize(&ResponseMessageType {
            rpc_id: 0,
            msg: crate::rpc::AnywhereRPCResponse::Canonicalize { res: "".into() },
        })
        .unwrap();

        let _: ResponseMessageType = bincode::deserialize(&ser).unwrap();
    }
}
