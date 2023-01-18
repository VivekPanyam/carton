//! This module implements FD passing between processes and channels/unix streams on top

use std::{
    collections::{HashMap, VecDeque},
    fmt::Debug,
    ops::Deref,
    os::fd::{FromRawFd, IntoRawFd, OwnedFd, RawFd},
    sync::atomic::AtomicU64,
};

use sendfd::{RecvWithFd, SendWithFd};
use serde::{de::DeserializeOwned, Serialize};
use tokio::{
    io::{AsyncRead, AsyncWrite},
    net::{UnixListener, UnixStream},
    sync::{mpsc, oneshot},
};

use super::{framed::frame, types::{FdId, ChannelId}};

/// The (internal) process type is defined by whether it uses `bind` or `connect`
#[derive(PartialEq, Eq)]
enum ProcessType {
    Primary,
    Secondary,
}

pub(crate) struct Comms {
    // Used to generate IDs for file descriptors
    fd_id_gen: AtomicU64,

    // The type of the current process
    process_type: ProcessType,

    // Outgoing file descriptors
    outgoing_tx: mpsc::Sender<(FdId, RawFd)>,

    // A queue used to register callbacks when we get a certain FD
    register_callbacks_tx: mpsc::Sender<(FdId, oneshot::Sender<RawFd>)>,
}

impl Comms {
    /// Connect to a unix domain socket given a path
    pub async fn connect(path: &std::path::Path) -> Self {
        let stream = UnixStream::connect(path).await.unwrap();

        let (outgoing_tx, outgoing_rx) = mpsc::channel(32);
        let (register_callbacks_tx, register_callbacks_rx) = mpsc::channel(32);

        // Handle messaging
        tokio::spawn(Comms::handle_stream(
            stream,
            outgoing_rx,
            register_callbacks_rx,
        ));

        Self {
            process_type: ProcessType::Secondary,
            fd_id_gen: (ChannelId::NUM_RESERVED_IDS as u64).into(), // Some IDs are reserved
            outgoing_tx,
            register_callbacks_tx,
        }
    }

    /// Initialize comms bound to the provided path
    pub async fn bind(bind_path: &std::path::Path) -> Self {
        let listener = UnixListener::bind(bind_path).unwrap();

        let (outgoing_tx, outgoing_rx) = mpsc::channel(32);
        let (register_callbacks_tx, register_callbacks_rx) = mpsc::channel(32);

        tokio::spawn(async move {
            // Wait for a connection
            let stream = match listener.accept().await {
                Ok((stream, _)) => stream,
                Err(e) => panic!("Error when connecting: {}", e),
            };

            // Handle messaging
            Comms::handle_stream(stream, outgoing_rx, register_callbacks_rx).await;
        });

        Self {
            process_type: ProcessType::Primary,
            fd_id_gen: (ChannelId::NUM_RESERVED_IDS as u64).into(), // Some IDs are reserved
            outgoing_tx,
            register_callbacks_tx,
        }
    }

    // Handle messaging on a UnixStream and callbacks in the local process
    async fn handle_stream(
        stream: UnixStream,
        mut outgoing_rx: mpsc::Receiver<(FdId, RawFd)>,
        mut register_callbacks_rx: mpsc::Receiver<(FdId, oneshot::Sender<RawFd>)>,
    ) {
        // Split into read and write
        let (read_stream, write_stream) = stream.into_split();

        // Spawn a task to write outgoing fds
        tokio::spawn(async move {
            let ws: &UnixStream = write_stream.as_ref();
            while let Some((id, fd)) = outgoing_rx.recv().await {
                let fds = [fd];
                let bytes = id.0.to_le_bytes();

                // Wait until writable and then write
                ws.writable().await.unwrap();
                ws.send_with_fd(&bytes, &fds).unwrap();
            }
        });

        let (incoming_tx, mut incoming_rx) = mpsc::channel(32);

        // Spawn a task to handle incoming fds
        tokio::spawn(async move {
            let rs: &UnixStream = read_stream.as_ref();

            let mut fd_queue: VecDeque<RawFd> = VecDeque::new();
            let mut id_queue: VecDeque<FdId> = VecDeque::new();

            loop {
                let mut bytes = [0u8; 8];
                let mut fds = [0; 1];

                // Wait until we can read
                rs.readable().await.unwrap();
                let (num_bytes, num_fds) = rs.recv_with_fd(&mut bytes, &mut fds).unwrap();

                if num_bytes != 0 {
                    assert_eq!(
                        num_bytes, 8,
                        "Got an unexpected number of bytes in FD recv code: {num_bytes}"
                    );

                    id_queue.push_back(FdId(u64::from_le_bytes(bytes)));
                }

                if num_fds != 0 {
                    assert_eq!(
                        num_fds, 1,
                        "Got an unexpected number of fds in FD recv code: {num_fds}"
                    );

                    fd_queue.push_back(fds[0]);
                }

                // Write them out
                if !fd_queue.is_empty() && !id_queue.is_empty() {
                    if incoming_tx
                        .send((id_queue.pop_front().unwrap(), fd_queue.pop_front().unwrap()))
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
            }
        });

        // Now handle local callbacks for FDs
        let mut waiting = HashMap::new();
        let mut received = HashMap::new();

        loop {
            let mut callback_req = None;
            let mut incoming_data = None;
            tokio::select! {
                cr = register_callbacks_rx.recv() => callback_req = cr,
                ic = incoming_rx.recv() => incoming_data = ic,
            }

            if let Some((requested_id, callback)) = callback_req {
                // Check if we already have the requested id
                if let Some(fd) = received.remove(&requested_id) {
                    callback.send(fd);
                } else {
                    // Put the callback in waiting
                    waiting.insert(requested_id, callback);
                }
            }

            if let Some((fd_id, fd)) = incoming_data {
                // Something was already waiting on this
                if let Some(callback) = waiting.remove(&fd_id) {
                    callback.send(fd);
                } else {
                    // Put it in received
                    received.insert(fd_id, fd);
                }
            }
        }
    }

    // Send an FD and return the id that can be used to access it on the other end
    pub(crate) async fn send_fd(&self, fd: RawFd) -> FdId {
        // id_gen * 2 for the primary process
        // id_gen * 2 + 1 for the secondary process
        // This gives us distinct id spaces
        let mut id = self
            .fd_id_gen
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            * 2;

        if self.process_type == ProcessType::Secondary {
            id += 1;
        }

        let id = FdId(id);
        self.outgoing_tx.send((id, fd)).await;
        id
    }

    // Get a file descriptor by ID or wait for it
    pub(crate) async fn wait_for_fd(&self, fd_id: FdId) -> RawFd {
        let (tx, rx) = oneshot::channel();

        self.register_callbacks_tx.send((fd_id, tx)).await;

        rx.await.unwrap()
    }

    /// Create a bidirectional stream and send one half of it to the other process
    /// Returns the stream
    async fn create_bidi_stream(&self, id: FdId) -> UnixStream {
        let (one, two) = UnixStream::pair().unwrap();
        let fd = two.into_std().unwrap().into_raw_fd();
        self.outgoing_tx.send((id, fd)).await;
        one
    }

    /// A bidirectional stream
    /// Note: this can only be called once per channel id
    async fn get_raw_channel(
        &self,
        channel_id: ChannelId,
    ) -> (impl AsyncRead, impl AsyncWrite) {
        let id = FdId(channel_id as u64);

        let stream = if self.process_type == ProcessType::Primary {
            // Create a bidirectional stream
            self.create_bidi_stream(id).await
        } else {
            // Wait for the channel created on the other end
            let fd = self.wait_for_fd(id).await;

            let owned = unsafe { OwnedFd::from_raw_fd(fd) };
            let std_stream = std::os::unix::net::UnixStream::from(owned);
            UnixStream::from_std(std_stream).unwrap()
        };

        // Split into read and write
        let (read_stream, write_stream) = stream.into_split();

        (read_stream, write_stream)
    }

    /// A framed transport that can transport serializable things on top of a bidirectional stream.
    /// Note: this can only be called once per channel id
    pub async fn get_channel<T, U>(
        &self,
        channel_id: ChannelId,
    ) -> (mpsc::Sender<T>, mpsc::Receiver<U>)
    where
        T: Debug + Serialize + Send + 'static,
        U: Debug + DeserializeOwned + Send + 'static,
    {
        let (read_stream, write_stream) = self.get_raw_channel(channel_id).await;
        frame(read_stream, write_stream).await
    }
}

/// A comms instance that "owns" the bootstrap unix domain socket
/// (and will delete it on drop)
pub(crate) struct OwnedComms {
    // A folder that stores the UDS we communicate using
    tempdir: tempfile::TempDir,

    // The comms layer
    comms: Comms,
}

impl OwnedComms {
    /// Returns Self and the bootstrap path for the other process to connect to
    pub(crate) async fn new() -> (Self, std::path::PathBuf) {
        // Create a UDS in a temp dir
        let tempdir = tempfile::tempdir().unwrap();
        let bind_path = tempdir.path().join("bootstrap");

        (
            Self {
                tempdir,
                comms: Comms::bind(bind_path.as_path()).await,
            },
            bind_path,
        )
    }
}

impl Deref for OwnedComms {
    type Target = Comms;

    fn deref(&self) -> &Self::Target {
        &self.comms
    }
}
