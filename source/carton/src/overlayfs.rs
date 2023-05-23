use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
};

use async_trait::async_trait;
use lunchbox::{
    path::PathBuf,
    types::{
        DirEntry, HasFileType, MaybeSend, MaybeSync, Metadata, PathType, ReadDir, ReadDirPoller,
        ReadableFile,
    },
    ReadableFileSystem,
};
use pin_project::pin_project;
use tokio::io::AsyncRead;

/// OverlayFS takes two filesystems and overlays the second on top of the first
/// It only supports readable filesystems
/// Each of the ops are implemented as below;
/// - `open`:
///     Try the top FS and then the bottom one on failure
/// - `canonicalize`:
///     If it's a file on the top filesystem, call through to canonicalize there
///     If it's a file on the bottom filesystem, call through to canonicalize there
///     Else use path_clean to normalize the path
/// - `metadata`:
///     Try the top FS and then the bottom one on failure
/// - `read`:
///     Try the top FS and then the bottom one on failure
/// - `read_dir`:
///     Call read_dir on both filesystems and merge the result
/// - `read_link`:
///     Try the top FS and then the bottom one on failure
/// - `read_to_string`:
///     Try the top FS and then the bottom one on failure
/// - `symlink_metadata`:
///     Try the top FS and then the bottom one on failure
pub(crate) struct OverlayFS<B, T> {
    bottom: Arc<B>,
    top: Arc<T>,
}

impl<B, T> OverlayFS<B, T> {
    pub fn new(bottom: Arc<B>, top: Arc<T>) -> Self {
        Self { bottom, top }
    }
}

/// The filetype for OverlayFS
#[pin_project(project = EnumProj)]
pub(crate) enum OverlayFile<B, T> {
    Bottom(#[pin] B),
    Top(#[pin] T),
}

impl<B: AsyncRead, T: AsyncRead> AsyncRead for OverlayFile<B, T> {
    fn poll_read(
        self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        match self.project() {
            EnumProj::Bottom(v) => v.poll_read(cx, buf),
            EnumProj::Top(v) => v.poll_read(cx, buf),
        }
    }
}

#[cfg_attr(target_family = "wasm", async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait)]
impl<B, T> ReadableFile for OverlayFile<B, T>
where
    B: ReadableFile + MaybeSend + MaybeSync,
    T: ReadableFile + MaybeSend + MaybeSync,
{
    async fn metadata(&self) -> std::io::Result<Metadata> {
        match self {
            OverlayFile::Bottom(v) => v.metadata().await,
            OverlayFile::Top(v) => v.metadata().await,
        }
    }

    async fn try_clone(&self) -> std::io::Result<Self> {
        match self {
            OverlayFile::Bottom(v) => v.try_clone().await.map(Self::Bottom),
            OverlayFile::Top(v) => v.try_clone().await.map(Self::Top),
        }
    }
}

impl<B: HasFileType, T: HasFileType> HasFileType for OverlayFS<B, T> {
    type FileType = OverlayFile<B::FileType, T::FileType>;
}

#[cfg_attr(target_family = "wasm", async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait)]
impl<B, T> ReadableFileSystem for OverlayFS<B, T>
where
    B: ReadableFileSystem + MaybeSend + MaybeSync,
    T: ReadableFileSystem + MaybeSend + MaybeSync,
    B::FileType: ReadableFile + MaybeSend + MaybeSync,
    T::FileType: ReadableFile + MaybeSend + MaybeSync,
    B::ReadDirPollerType: MaybeSend,
    T::ReadDirPollerType: MaybeSend,
{
    async fn open(&self, path: impl PathType) -> std::io::Result<Self::FileType>
    where
        Self::FileType: ReadableFile,
    {
        let p = path.as_ref();

        fallthrough(
            || async { self.top.open(p).await.map(OverlayFile::Top) },
            || async { self.bottom.open(p).await.map(OverlayFile::Bottom) },
        )
        .await
    }

    async fn canonicalize(&self, path: impl PathType) -> std::io::Result<PathBuf> {
        let p = path.as_ref();

        if self.top.metadata(p).await.is_ok() {
            // It exists on the top fs
            self.top.canonicalize(p).await
        } else if self.bottom.metadata(p).await.is_ok() {
            // It exists on the bottom fs
            self.bottom.canonicalize(p).await
        } else {
            // We'll use path_clean to normalize the path
            Ok(path_clean::clean(p.as_str()).into())
        }
    }

    async fn metadata(&self, path: impl PathType) -> std::io::Result<Metadata> {
        let p = path.as_ref();

        fallthrough(
            || async { self.top.metadata(p).await },
            || async { self.bottom.metadata(p).await },
        )
        .await
    }

    async fn read(&self, path: impl PathType) -> std::io::Result<Vec<u8>> {
        let p = path.as_ref();

        fallthrough(
            || async { self.top.read(p).await },
            || async { self.bottom.read(p).await },
        )
        .await
    }

    type ReadDirPollerType = OverlayReadDirPoller;

    async fn read_dir(
        &self,
        path: impl PathType,
    ) -> std::io::Result<ReadDir<Self::ReadDirPollerType, Self>> {
        // This isn't super ideal because we're doing complete read_dir operations on both filesystems up front
        // There's probably a better way to do it as part of poll_next_entry, but this is fine for now
        let p = path.as_ref();

        // Map from path to Entry
        let mut entries = HashMap::new();

        let top_info = self.top.read_dir(p).await;
        let bottom_info = self.bottom.read_dir(p).await;

        // If both are errors, return an error
        // Note: we can't combine these two into one if statement because it's an unstable feature
        if top_info.is_err() {
            if let Err(e) = bottom_info {
                // Return the bottom err because we do that with all the other methods
                // TODO: Does this choice make sense?
                return Err(e);
            }
        }

        // Merge in the bottom entries
        if let Ok(mut dir) = bottom_info {
            while let Some(entry) = dir.next_entry().await? {
                let p = entry.path();
                entries.insert(
                    p.clone(),
                    Entry {
                        file_name: entry.file_name(),
                        path: p,
                    },
                );
            }
        }

        // Merge in the top entries (which should overwrite any duplicates on the bottom fs)
        if let Ok(mut dir) = top_info {
            while let Some(entry) = dir.next_entry().await? {
                let p = entry.path();
                entries.insert(
                    p.clone(),
                    Entry {
                        file_name: entry.file_name(),
                        path: p,
                    },
                );
            }
        }

        let poller = OverlayReadDirPoller {
            entries: entries.into_iter().map(|(_, v)| v).collect(),
        };
        Ok(ReadDir::new(poller, self))
    }

    async fn read_link(&self, path: impl PathType) -> std::io::Result<PathBuf> {
        let p = path.as_ref();

        fallthrough(
            || async { self.top.read_link(p).await },
            || async { self.bottom.read_link(p).await },
        )
        .await
    }

    async fn read_to_string(&self, path: impl PathType) -> std::io::Result<String> {
        let p = path.as_ref();

        fallthrough(
            || async { self.top.read_to_string(p).await },
            || async { self.bottom.read_to_string(p).await },
        )
        .await
    }

    async fn symlink_metadata(&self, path: impl PathType) -> std::io::Result<Metadata> {
        let p = path.as_ref();

        fallthrough(
            || async { self.top.symlink_metadata(p).await },
            || async { self.bottom.symlink_metadata(p).await },
        )
        .await
    }
}

pub(crate) struct OverlayReadDirPoller {
    entries: VecDeque<Entry>,
}

// Subset of DirEntry
struct Entry {
    file_name: String,
    path: PathBuf,
}

impl<F> ReadDirPoller<F> for OverlayReadDirPoller
where
    F: ReadableFileSystem,
    F::FileType: ReadableFile,
{
    fn poll_next_entry<'a>(
        &mut self,
        _cx: &mut std::task::Context<'_>,
        fs: &'a F,
    ) -> std::task::Poll<std::io::Result<Option<lunchbox::types::DirEntry<'a, F>>>> {
        std::task::Poll::Ready(Ok(self
            .entries
            .pop_front()
            .map(|v| DirEntry::new(fs, v.file_name, v.path))))
    }
}

async fn fallthrough<T, U, R, Fut, Fut2>(t: T, u: U) -> std::io::Result<R>
where
    T: FnOnce() -> Fut,
    U: FnOnce() -> Fut2,
    Fut: std::future::Future<Output = std::io::Result<R>>,
    Fut2: std::future::Future<Output = std::io::Result<R>>,
{
    if let Ok(v) = t().await {
        return Ok(v);
    }

    u().await
}
