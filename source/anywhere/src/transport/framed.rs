use std::sync::Arc;

use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader, BufWriter},
    sync::mpsc,
};

use crate::{
    rpc::{AnywhereRPCClient, AnywhereRPCServer, MaybeRead, MaybeWrite, MaybeSeek},
    types::AnywhereFS,
};

use super::{
    serde::{SerdeTransportClient, SerdeTransportServer},
    Transport,
};

/// Sends length prefixed frames over something that implements [`AsyncRead`] and [`AsyncWrite`]
pub struct FramedTransportClient {
    send_queue: mpsc::Sender<crate::rpc::MessageType>,
}

impl FramedTransportClient {
    pub async fn new<S, R>(send: S, recv: R) -> Self
    where
        S: 'static + AsyncWrite + Send + Unpin,
        R: 'static + AsyncRead + Send + Unpin,
    {
        let serde_transport = Arc::new(SerdeTransportClient::new());
        let (tx, mut rx) = mpsc::channel(32);

        // Send items
        let s = serde_transport.clone();
        let mut writer = BufWriter::new(send);
        tokio::spawn(async move {
            while let Some(item) = rx.recv().await {
                let msg = s.transform_req(item);
                let encoded = bincode::serialize(&msg).unwrap();

                // Write the data
                writer.write_u64(encoded.len() as _).await.unwrap();
                writer.write_all(&encoded).await.unwrap();
            }
        });

        // Recv responses
        let mut reader = BufReader::new(recv);
        tokio::spawn(async move {
            loop {
                let len = reader.read_u64().await.unwrap() as usize;

                let mut data = vec![0u8; len];
                reader.read_exact(data.as_mut_slice()).await.unwrap();

                // Handle responses
                serde_transport.on_res(bincode::deserialize(data.as_slice()).unwrap());
            }
        });

        Self { send_queue: tx }
    }

    pub async fn send(&self, msg: crate::rpc::MessageType) {
        self.send_queue.send(msg).await.unwrap();
    }

    pub async fn handle_queue(self, mut q: mpsc::Receiver<crate::rpc::MessageType>) {
        tokio::spawn(async move {
            while let Some(item) = q.recv().await {
                self.send(item).await;
            }
        });
    }
}

pub async fn connect<S, R, const WRITABLE: bool, const SEEKABLE: bool>(
    send: S,
    recv: R,
) -> std::io::Result<AnywhereFS<WRITABLE, SEEKABLE>>
where
    S: 'static + AsyncWrite + Send + Unpin,
    R: 'static + AsyncRead + Send + Unpin,
{
    // Create a queue
    let (tx, rx) = mpsc::channel(32);

    // Handle the queue with a framed transport
    FramedTransportClient::new(send, recv)
        .await
        .handle_queue(rx)
        .await;

    // Create a client
    AnywhereRPCClient::new(tx).try_to_fs().await
}

pub(crate) async fn serve_fs<S, R, T, A: MaybeRead<T>, B: MaybeWrite<T>, C: MaybeSeek<T>>(fs: AnywhereRPCServer<T, A, B, C>, send: S, recv: R)
where
    S: AsyncWrite + Send + Unpin,
    R: AsyncRead + Send + Unpin,
    // T: Readable + MaybeWritable + MaybeSeekable,
{
    let s = SerdeTransportServer::new(fs);

    let mut writer = BufWriter::new(send);
    let mut reader = BufReader::new(recv);

    loop {
        // Get a request
        let len = reader.read_u64().await.unwrap() as usize;
        let mut data = vec![0u8; len];
        reader.read_exact(data.as_mut_slice()).await.unwrap();

        // Deserialize the request
        let req = bincode::deserialize(data.as_slice()).unwrap();

        // Handle the request and get a response
        let res = s.handle_request(req).await;

        // Serialize the response
        let encoded = bincode::serialize(&res).unwrap();

        // Write the data
        writer.write_u64(encoded.len() as _).await.unwrap();
        writer.write_all(&encoded).await.unwrap();
    }
}

pub struct FramedTransport;
impl Transport for FramedTransport {
    type Ret<T, A, B, C> = FramedTransportServer<T, A, B, C>;

    fn new<T, A, B, C>(inner: AnywhereRPCServer<T, A, B, C>) -> Self::Ret<T, A, B, C> {
        FramedTransportServer { inner }
    }
}

pub struct FramedTransportServer<T, A, B, C> {
    inner: crate::rpc::AnywhereRPCServer<T, A, B, C>,
}

impl<T, A: MaybeRead<T>, B: MaybeWrite<T>, C: MaybeSeek<T>> FramedTransportServer<T, A, B, C>
{
    /// Serves a filesystem by handling length prefixed frames over something that implements [`AsyncRead`] and [`AsyncWrite`]
    pub async fn serve<S, R>(self, send: S, recv: R)
    where
        S: AsyncWrite + Send + Unpin,
        R: AsyncRead + Send + Unpin,
    {
        serve_fs(self.inner, send, recv).await
    }
}
