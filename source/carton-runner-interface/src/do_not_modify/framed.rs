// Copyright 2023 Vivek Panyam
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! A framed transport on top of an [`AsyncRead`] and [`AsyncWrite`] pair

use std::{fmt::Debug, io::ErrorKind};

use serde::{de::DeserializeOwned, Serialize};
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader, BufWriter},
    sync::mpsc,
};

use crate::{do_spawn, MaybeSend};

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
            let size = match br.read_u64().await {
                Ok(s) => s as usize,
                Err(e) if e.kind() == ErrorKind::UnexpectedEof => break,
                Err(e) => panic!("Got unexpected error: {:#?}", e),
            };

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
                    match bw.flush().await {
                        Ok(_) => {}
                        Err(e) if e.kind() == ErrorKind::BrokenPipe => {
                            // Disconnected
                            break;
                        }
                        e => e.unwrap(),
                    }

                    // Blocking wait for new things to send
                    match req_rx.recv().await {
                        Some(item) => item,
                        // Disconnected
                        None => break,
                    }
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
