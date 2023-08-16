//! Multiplex several streams of a single type on top of a single channel
//!
//! Note: This can be susceptible to head-of-line blocking so don't use it for things where that
//! can be a problem or cause deadlocks.

use std::sync::{atomic::AtomicU64, Arc};

use crate::do_not_modify::types::StreamID;
use dashmap::DashMap;
use tokio::sync::mpsc;

pub(crate) struct Multiplexer<T, U> {
    id_gen: AtomicU64,
    send: mpsc::Sender<(StreamID, T)>,

    callbacks: Arc<dashmap::DashMap<StreamID, mpsc::Sender<U>>>,
}

impl<T, U> Multiplexer<T, U>
where
    T: Send + 'static,
    U: Send + 'static,
{
    ///
    /// **IMPORTANT**: Be careful when modifying the signature because it can affect the wire protocol
    ///
    pub(crate) async fn new(
        send: mpsc::Sender<(StreamID, T)>,
        mut recv: mpsc::Receiver<(StreamID, U)>,
    ) -> Self {
        // Handle routing received messages
        let callbacks: Arc<dashmap::DashMap<StreamID, mpsc::Sender<U>>> = Arc::new(DashMap::new());
        let callbacks_clone = callbacks.clone();
        tokio::spawn(async move {
            while let Some((id, item)) = recv.recv().await {
                if let Some(callback) = callbacks_clone.get(&id) {
                    callback
                        .value()
                        .send(item)
                        .await
                        .map_err(|_| "send failed")
                        .unwrap();
                } else {
                    panic!(
                        "Multiplexer got message for stream with unknown id {}",
                        id.0
                    );
                }
            }
        });

        Self {
            send,
            callbacks,
            id_gen: 0.into(),
        }
    }

    pub(crate) async fn get_new_stream(&self) -> (mpsc::Sender<T>, mpsc::Receiver<U>, StreamID) {
        let id = self
            .id_gen
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let id = StreamID(id);

        let (tx, rx) = self.get_stream_for_id(id).await;

        (tx, rx, id)
    }

    /// Note: this should only be called once per ID
    /// Generally, the same process should not use both get_new_stream and get_stream_for_id because
    /// they can stomp on each other
    pub(crate) async fn get_stream_for_id(
        &self,
        id: StreamID,
    ) -> (mpsc::Sender<T>, mpsc::Receiver<U>) {
        let (send_tx, mut send_rx) = mpsc::channel(32);
        let (recv_tx, recv_rx) = mpsc::channel(32);

        // Handle receiving
        self.callbacks.insert(id, recv_tx);

        // Handle sending
        let send = self.send.clone();
        tokio::spawn(async move {
            while let Some(item) = send_rx.recv().await {
                send.send((id, item))
                    .await
                    .map_err(|_| "send failed")
                    .unwrap();
            }
        });

        (send_tx, recv_rx)
    }
}
