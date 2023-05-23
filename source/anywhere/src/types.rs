use crate::{read_dir_ops::SerializedDirEntry, rpc::AnywhereRPCClient};
use async_trait::async_trait;
use futures::FutureExt;
use lunchbox::{
    path::PathBuf,
    types::{
        HasFileType, MaybeSend, Metadata, OpenOptions, PathType, Permissions, ReadDir,
        ReadDirPoller,
    },
};

use std::{collections::VecDeque, io::Result, pin::Pin, sync::Arc};
use tokio::io::{AsyncRead, AsyncSeek, AsyncWrite};

// The RPC path on the wire
pub type RPCPath = String;

// A handle to a file
pub type FileHandle = u64;

pub enum FileSystem {
    /// A readonly filesystem with a file type that implements AsyncRead
    Read(ReadOnlyFS),

    /// A readonly filesystem with a file type that implements AsyncRead + AsyncSeek
    ReadSeek(ReadOnlySeekableFS),

    /// A read/write filesystem with a file type that implements AsyncRead + AsyncWrite
    ReadWrite(ReadWriteFS),

    /// A read/write filesystem with a file type that implements AsyncRead + AsyncWrite + AsyncSeek
    ReadWriteSeek(ReadWriteSeekableFS),
}

pub type ReadOnlyFS = AnywhereFS<false, false>;
pub type ReadOnlySeekableFS = AnywhereFS<false, true>;
pub type ReadWriteFS = AnywhereFS<true, false>;
pub type ReadWriteSeekableFS = AnywhereFS<true, true>;

#[derive(Default)]
struct FileFutures {
    seek_fut:
        Option<Pin<Box<dyn std::future::Future<Output = std::io::Result<u64>> + Send + Sync>>>,

    read_fut:
        Option<Pin<Box<dyn std::future::Future<Output = std::io::Result<Vec<u8>>> + Send + Sync>>>,

    write_fut:
        Option<Pin<Box<dyn std::future::Future<Output = std::io::Result<usize>> + Send + Sync>>>,
    flush_fut:
        Option<Pin<Box<dyn std::future::Future<Output = std::io::Result<()>> + Send + Sync>>>,
    shutdown_fut:
        Option<Pin<Box<dyn std::future::Future<Output = std::io::Result<()>> + Send + Sync>>>,
}

pub struct AnywhereFile<const IS_WRITABLE: bool, const IS_SEEKABLE: bool> {
    handle: FileHandle,
    client: Arc<AnywhereRPCClient>,

    fut: FileFutures,
}

/// Implement AsyncRead for `AnywhereFile`s
impl<const W: bool, const S: bool> AsyncRead for AnywhereFile<W, S> {
    #[tracing::instrument(skip(self, cx, buf), fields(buf_remaining = buf.remaining()))]
    fn poll_read(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        // Create a new future if we need to
        if self.fut.read_fut.is_none() {
            let client = self.client.clone();
            let handle = self.handle;
            let max_num_bytes = buf.remaining();
            self.fut.read_fut = Some(Box::pin(async move {
                client.read_bytes(handle, max_num_bytes as _).await
            }));
        }

        // Check if the future is ready
        match self.fut.read_fut.as_mut().unwrap().poll_unpin(cx) {
            std::task::Poll::Ready(res) => {
                self.fut.read_fut = None;

                let res = res.map(|item| {
                    buf.put_slice(&item);
                });

                std::task::Poll::Ready(res)
            }
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}

#[cfg_attr(target_family = "wasm", async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait)]
impl<const W: bool, const S: bool> lunchbox::types::ReadableFile for AnywhereFile<W, S> {
    async fn metadata(&self) -> Result<Metadata> {
        self.client.file_metadata(self.handle).await
    }

    async fn try_clone(&self) -> Result<Self> {
        let handle = self.client.file_try_clone(self.handle).await?;
        Ok(AnywhereFile {
            handle,
            client: self.client.clone(),
            fut: Default::default(),
        })
    }
}

/// Implement AsyncWrite for writable `AnywhereFile`s
impl<const S: bool> AsyncWrite for AnywhereFile<true, S> {
    fn poll_write(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<std::result::Result<usize, std::io::Error>> {
        // Create a new future if we need to
        if self.fut.write_fut.is_none() {
            let client = self.client.clone();
            let handle = self.handle;
            let buf = buf.to_vec();
            self.fut.write_fut = Some(Box::pin(
                async move { client.write_data(handle, buf).await },
            ));
        }

        // Check if the future is ready
        match self.fut.write_fut.as_mut().unwrap().poll_unpin(cx) {
            std::task::Poll::Ready(res) => {
                self.fut.write_fut = None;

                std::task::Poll::Ready(res)
            }
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }

    fn poll_flush(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::result::Result<(), std::io::Error>> {
        // Create a new future if we need to
        if self.fut.flush_fut.is_none() {
            let client = self.client.clone();
            let handle = self.handle;
            self.fut.flush_fut = Some(Box::pin(async move { client.write_flush(handle).await }));
        }

        // Check if the future is ready
        match self.fut.flush_fut.as_mut().unwrap().poll_unpin(cx) {
            std::task::Poll::Ready(res) => {
                self.fut.flush_fut = None;

                std::task::Poll::Ready(res)
            }
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }

    fn poll_shutdown(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::result::Result<(), std::io::Error>> {
        // Create a new future if we need to
        if self.fut.shutdown_fut.is_none() {
            let client = self.client.clone();
            let handle = self.handle;
            self.fut.shutdown_fut =
                Some(Box::pin(async move { client.write_shutdown(handle).await }));
        }

        // Check if the future is ready
        match self.fut.shutdown_fut.as_mut().unwrap().poll_unpin(cx) {
            std::task::Poll::Ready(res) => {
                self.fut.shutdown_fut = None;

                std::task::Poll::Ready(res)
            }
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}

#[cfg_attr(target_family = "wasm", async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait)]
impl<const S: bool> lunchbox::types::WritableFile for AnywhereFile<true, S> {
    async fn sync_all(&self) -> Result<()> {
        self.client.file_sync_all(self.handle).await
    }

    async fn sync_data(&self) -> Result<()> {
        self.client.file_sync_data(self.handle).await
    }

    async fn set_len(&self, size: u64) -> Result<()> {
        self.client.file_set_len(self.handle, size).await
    }

    async fn set_permissions(&self, perm: Permissions) -> Result<()> {
        self.client.file_set_permissions(self.handle, perm).await
    }
}

/// Implement AsyncSeek for seekable `AnywhereFile`s
impl<const W: bool> AsyncSeek for AnywhereFile<W, true> {
    fn start_seek(
        self: std::pin::Pin<&mut Self>,
        position: std::io::SeekFrom,
    ) -> std::io::Result<()> {
        let client = self.client.clone();
        let handle = self.handle;
        self.get_mut().fut.seek_fut =
            Some(Box::pin(async move { client.seek(handle, position).await }));

        Ok(())
    }

    fn poll_complete(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<u64>> {
        match self.fut.seek_fut.as_mut().unwrap().poll_unpin(cx) {
            std::task::Poll::Ready(res) => {
                // This future is done so lets clear seek_fut and return the result
                self.fut.seek_fut = None;

                std::task::Poll::Ready(res)
            }
            std::task::Poll::Pending => std::task::Poll::Pending,
        }
    }
}

pub struct AnywhereFS<const W: bool, const S: bool> {
    pub(crate) client: Arc<AnywhereRPCClient>,
}

impl<const W: bool, const S: bool> HasFileType for AnywhereFS<W, S> {
    type FileType = AnywhereFile<W, S>;
}

#[cfg_attr(target_family = "wasm", async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait)]
impl<const W: bool, const S: bool> lunchbox::ReadableFileSystem for AnywhereFS<W, S> {
    // Open a file
    async fn open(&self, path: impl PathType) -> Result<Self::FileType> {
        self.client
            .open_file(path.convert())
            .await
            .convert(&self.client)
    }

    // These are directly based on tokio::fs::...
    async fn canonicalize(&self, path: impl PathType) -> Result<PathBuf> {
        self.client.canonicalize(path.convert()).await
    }

    async fn metadata(&self, path: impl PathType) -> Result<Metadata> {
        self.client.metadata(path.convert()).await
    }

    async fn read(&self, path: impl PathType) -> Result<Vec<u8>> {
        self.client.read(path.convert()).await
    }

    type ReadDirPollerType = AnywhereFSReadDirPoller;
    async fn read_dir(
        &self,
        path: impl PathType,
    ) -> Result<ReadDir<Self::ReadDirPollerType, Self>> {
        let out = self.client.read_dir_wrapper(path.convert()).await?;

        Ok(ReadDir::new(
            AnywhereFSReadDirPoller {
                entries: out.into(),
            },
            self,
        ))
    }

    async fn read_link(&self, path: impl PathType) -> Result<PathBuf> {
        self.client.read_link(path.convert()).await
    }

    async fn read_to_string(&self, path: impl PathType) -> Result<String> {
        self.client.read_to_string(path.convert()).await
    }

    async fn symlink_metadata(&self, path: impl PathType) -> Result<Metadata> {
        self.client.symlink_metadata(path.convert()).await
    }
}

pub struct AnywhereFSReadDirPoller {
    entries: VecDeque<SerializedDirEntry>,
}

impl<const W: bool, const S: bool> ReadDirPoller<AnywhereFS<W, S>> for AnywhereFSReadDirPoller {
    fn poll_next_entry<'a>(
        &mut self,
        _cx: &mut std::task::Context<'_>,
        fs: &'a AnywhereFS<W, S>,
    ) -> std::task::Poll<std::io::Result<Option<lunchbox::types::DirEntry<'a, AnywhereFS<W, S>>>>>
    {
        std::task::Poll::Ready(Ok(self.entries.pop_front().map(|item| {
            lunchbox::types::DirEntry::new(fs, item.file_name, item.path.into())
        })))
    }
}

#[cfg_attr(target_family = "wasm", async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait)]
impl<const S: bool> lunchbox::WritableFileSystem for AnywhereFS<true, S> {
    // Create a file
    async fn create(&self, path: impl PathType) -> Result<Self::FileType> {
        self.client
            .create_file(path.convert())
            .await
            .convert(&self.client)
    }

    async fn open_with_opts(
        &self,
        opts: &OpenOptions,
        path: impl PathType,
    ) -> Result<Self::FileType> {
        self.client
            .open_file_with_opts(opts.clone(), path.convert())
            .await
            .convert(&self.client)
    }

    // These are directly based on tokio::fs::...
    async fn copy(&self, from: impl PathType, to: impl PathType) -> Result<u64> {
        self.client.copy(from.convert(), to.convert()).await
    }

    async fn create_dir(&self, path: impl PathType) -> Result<()> {
        self.client.create_dir(path.convert()).await
    }

    async fn create_dir_all(&self, path: impl PathType) -> Result<()> {
        self.client.create_dir_all(path.convert()).await
    }

    async fn hard_link(&self, src: impl PathType, dst: impl PathType) -> Result<()> {
        self.client.hard_link(src.convert(), dst.convert()).await
    }

    async fn remove_dir(&self, path: impl PathType) -> Result<()> {
        self.client.remove_dir(path.convert()).await
    }

    async fn remove_dir_all(&self, path: impl PathType) -> Result<()> {
        self.client.remove_dir_all(path.convert()).await
    }

    async fn remove_file(&self, path: impl PathType) -> Result<()> {
        self.client.remove_file(path.convert()).await
    }

    async fn rename(&self, from: impl PathType, to: impl PathType) -> Result<()> {
        self.client.rename(from.convert(), to.convert()).await
    }

    async fn set_permissions(&self, path: impl PathType, perm: Permissions) -> Result<()> {
        self.client
            .set_permissions(path.convert(), perm.convert())
            .await
    }

    async fn symlink(&self, src: impl PathType, dst: impl PathType) -> Result<()> {
        self.client.symlink(src.convert(), dst.convert()).await
    }

    async fn write(
        &self,
        path: impl PathType,
        contents: impl AsRef<[u8]> + MaybeSend,
    ) -> Result<()> {
        self.client.write(path.convert(), contents.convert()).await
    }
}

trait TypeConversion<T> {
    fn convert(self) -> T;
}

impl<T: PathType> TypeConversion<String> for T {
    fn convert(self) -> String {
        self.as_ref().as_str().to_owned()
    }
}

impl<T: AsRef<[u8]>> TypeConversion<Vec<u8>> for T {
    fn convert(self) -> Vec<u8> {
        self.as_ref().to_vec()
    }
}

trait AutoImplConversion {}

impl AutoImplConversion for Permissions {}
impl AutoImplConversion for OpenOptions {}

impl<T: AutoImplConversion> TypeConversion<T> for T {
    fn convert(self) -> T {
        self
    }
}

trait TypeConversionWithContext<T> {
    fn convert(self, client: &Arc<AnywhereRPCClient>) -> T;
}

impl<const W: bool, const S: bool> TypeConversionWithContext<Result<AnywhereFile<W, S>>>
    for std::io::Result<FileHandle>
{
    fn convert(self, client: &Arc<AnywhereRPCClient>) -> Result<AnywhereFile<W, S>> {
        let handle = self?;
        Ok(AnywhereFile {
            handle,
            client: client.clone(),
            fut: Default::default(),
        })
    }
}
