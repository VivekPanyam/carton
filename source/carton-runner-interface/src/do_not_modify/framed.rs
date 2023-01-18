//! A framed transport on top of an [`AsyncRead`] and [`AsyncWrite`] pair

use std::fmt::Debug;

use serde::{de::DeserializeOwned, Serialize};
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader, BufWriter},
    sync::mpsc,
};

use crate::{MaybeSend, do_spawn};

/// Send and recv length-prefixed serialized structs on an [`AsyncRead`] and [`AsyncWrite`] pair
pub(crate) async fn framed_transport<T, U, R, W>(
    read_stream: R,
    write_stream: W,
    mut req_rx: mpsc::Receiver<T>,
    res_tx: mpsc::Sender<U>,
) where
    R: AsyncRead + Unpin + MaybeSend + 'static,
    W: AsyncWrite + Unpin + MaybeSend + 'static,
    T: Debug + Serialize + Send + 'static,
    U: Debug + DeserializeOwned + Send + 'static,
{
    // Spawn a task to handle reads
    do_spawn(async move {
        let mut br = BufReader::new(read_stream);

        loop {
            // Read the size and then read the data
            let size = br.read_u64().await.unwrap() as usize;

            let mut data = vec![0u8; size];
            br.read_exact(&mut data).await.unwrap();

            // TODO: offload this to a compute thread if it's too slow
            let response: U = bincode::deserialize(&data).unwrap();

            // Send the response
            res_tx.send(response).await.unwrap();
        }
    });

    // Handle writes
    do_spawn(async move {
        let mut bw = BufWriter::new(write_stream);
        loop {
            let item = match req_rx.try_recv() {
                Ok(item) => item,
                Err(mpsc::error::TryRecvError::Empty) => {
                    // Nothing to recv
                    // Flush the writer
                    bw.flush().await.unwrap();

                    // Blocking wait for new things to send
                    req_rx.recv().await.unwrap()
                }
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    // We're done
                    break;
                }
            };

            // Serialize and write size + data to the buffer
            // TODO: offload this to a compute thread if it's too slow
            let data = bincode::serialize(&item).unwrap();
            bw.write_u64(data.len() as _).await.unwrap();
            bw.write_all(&data).await.unwrap();
        }
    });
}

pub(crate) async fn frame<T, U, R, W>(
    read_stream: R,
    write_stream: W,
) -> (mpsc::Sender<T>, mpsc::Receiver<U>)
where
    R: AsyncRead + Unpin + MaybeSend + 'static,
    W: AsyncWrite + Unpin + MaybeSend + 'static,
    T: Debug + Serialize + Send + 'static,
    U: Debug + DeserializeOwned + Send + 'static,
{
    let (send, req_rx) = mpsc::channel(32);
    let (res_tx, recv) = mpsc::channel(32);

    // Spawns tasks
    framed_transport(read_stream, write_stream, req_rx, res_tx).await;

    (send, recv)
}
