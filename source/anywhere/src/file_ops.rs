use std::io::SeekFrom;

use crate::{
    rpc::ServerContext,
    types::{FileHandle, RPCPath},
};
use async_trait::async_trait;
use lunchbox::{
    types::{Metadata, OpenOptions, Permissions, ReadableFile, WritableFile},
    ReadableFileSystem, WritableFileSystem,
};
use tokio::io::{AsyncReadExt, AsyncSeek, AsyncSeekExt, AsyncWriteExt};

// This lets us do file operations "on" the filesystem directly using a token
#[async_trait]
pub(crate) trait ReadableFileOps: ReadableFileSystem
where
    Self::FileType: Send + Sync + ReadableFile + Unpin,
{
    // File IO
    async fn read_bytes(
        &self,
        context: &ServerContext<Self>,
        handle: FileHandle,
        max_num_bytes: u64,
    ) -> std::io::Result<Vec<u8>> {
        match context.open_files.get_mut(&handle) {
            Some(mut f) => {
                let mut buf = vec![0u8; max_num_bytes as _];
                f.read(&mut buf).await.map(|len| {
                    buf.truncate(len);
                    buf
                })
            }
            None => todo!(),
        }
    }

    // Read only file operations
    async fn file_metadata(
        &self,
        context: &ServerContext<Self>,
        handle: FileHandle,
    ) -> std::io::Result<Metadata> {
        match context.open_files.get(&handle) {
            Some(f) => f.metadata().await,
            None => todo!(),
        }
    }

    async fn file_try_clone(
        &self,
        context: &ServerContext<Self>,
        handle: FileHandle,
    ) -> std::io::Result<FileHandle> {
        match context.open_files.get(&handle) {
            Some(f) => f.try_clone().await.map(|new_file| {
                let new_handle = context
                    .file_handle_counter
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                context.open_files.insert(new_handle, new_file);
                new_handle
            }),
            None => todo!(),
        }
    }

    // Read only filesystem operations
    async fn open_file(
        &self,
        context: &ServerContext<Self>,
        path: RPCPath,
    ) -> std::io::Result<FileHandle> {
        self.open(path).await.map(|file| {
            let handle = context
                .file_handle_counter
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            context.open_files.insert(handle, file);
            handle
        })
    }
}

impl<T: ReadableFileSystem> ReadableFileOps for T where
    T::FileType: Send + Sync + ReadableFile + Unpin
{
}

#[async_trait]
pub(crate) trait WritableFileOps: WritableFileSystem where Self::FileType: Send + Sync + WritableFile + Unpin {
    // File IO
    async fn write_data(&self, context: &ServerContext<Self>, handle: FileHandle, buf: Vec<u8>) -> std::io::Result<usize> {
        let mut item = context.open_files.get_mut(&handle).unwrap();
        item.write(&buf).await
    }

    async fn write_flush(&self, context: &ServerContext<Self>, handle: FileHandle) -> std::io::Result<()> {
        let mut item = context.open_files.get_mut(&handle).unwrap();
        item.value_mut().flush().await
    }
    async fn write_shutdown(&self, context: &ServerContext<Self>, handle: FileHandle) -> std::io::Result<()> {
        let mut item = context.open_files.get_mut(&handle).unwrap();
        item.value_mut().shutdown().await
    }

    // Write file operations
    async fn file_sync_all(&self, context: &ServerContext<Self>, handle: FileHandle) -> std::io::Result<()> {
        let mut item = context.open_files.get_mut(&handle).unwrap();
        item.value_mut().sync_all().await
    }

    async fn file_sync_data(&self, context: &ServerContext<Self>, handle: FileHandle) -> std::io::Result<()> {
        let mut item = context.open_files.get_mut(&handle).unwrap();
        item.value_mut().sync_data().await
    }

    async fn file_set_len(&self, context: &ServerContext<Self>, handle: FileHandle, size: u64) -> std::io::Result<()> {
        let mut item = context.open_files.get_mut(&handle).unwrap();
        item.value_mut().set_len(size).await
    }

    async fn file_set_permissions(
        &self,
        context: &ServerContext<Self>, 
        handle: FileHandle,
        perm: Permissions,
    ) -> std::io::Result<()> {
        let mut item = context.open_files.get_mut(&handle).unwrap();
        item.value_mut().set_permissions(perm).await
    }

    // Write filesystem operations
    async fn create_file(&self, context: &ServerContext<Self>, path: RPCPath) -> std::io::Result<FileHandle> {
        self.create(path).await.map(|file| {
            let handle = context
                .file_handle_counter
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            context.open_files.insert(handle, file);
            handle
        })
    }
    async fn open_file_with_opts(
        &self,
        context: &ServerContext<Self>, 
        opts: OpenOptions,
        path: RPCPath,
    ) -> std::io::Result<FileHandle> {
        self.open_with_opts(&opts, path).await.map(|file| {
            let handle = context
                .file_handle_counter
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            context.open_files.insert(handle, file);
            handle
        })
    }
}

impl <T: WritableFileSystem> WritableFileOps for T where T::FileType: Send + Sync + WritableFile + Unpin {}

#[async_trait]
pub(crate) trait SeekableFileOps: ReadableFileSystem
where
    Self::FileType: Send + Sync + AsyncSeek + Unpin,
{
    // File IO
    async fn seek(
        &self,
        context: &ServerContext<Self>,
        handle: FileHandle,
        pos: SeekFrom,
    ) -> std::io::Result<u64> {
        let mut item = context.open_files.get_mut(&handle).unwrap();
        item.value_mut().seek(pos).await
    }
}

impl<T: ReadableFileSystem> SeekableFileOps for T where T::FileType: Send + Sync + AsyncSeek + Unpin {}
