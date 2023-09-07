use crate::error::{CartonError, Result};
use async_trait::async_trait;
use dashmap::DashMap;
use futures::TryStreamExt;
use lazy_static::lazy_static;
use lunchbox::{
    path::PathBuf,
    types::{
        DirEntry, FileType, HasFileType, Metadata, PathType, Permissions, ReadDir, ReadDirPoller,
        ReadableFile,
    },
    ReadableFileSystem,
};
use std::{
    collections::{HashMap, VecDeque},
    pin::Pin,
    task::Poll,
};
use tokio::io::AsyncRead;
use tokio_util::compat::FuturesAsyncReadCompatExt;

/// HTTPFile implements [`AsyncRead`] on top of an HTTP request
pub(crate) struct HTTPFile {
    client: reqwest::Client,
    info: FileInfo,
    file_len: u64,

    // The current position
    seek_pos: u64,

    state: RequestState,
}

enum RequestState {
    None,

    /// The current request we're waiting on (if any)
    #[cfg(target_family = "wasm")]
    Request(Pin<Box<dyn std::future::Future<Output = Box<dyn AsyncRead + Unpin>>>>),

    #[cfg(not(target_family = "wasm"))]
    Request(
        Pin<
            Box<
                dyn std::future::Future<Output = Box<dyn AsyncRead + Unpin + Send + Sync>>
                    + Send
                    + Sync,
            >,
        >,
    ),

    /// The current streaming response
    #[cfg(target_family = "wasm")]
    Response(Pin<Box<dyn AsyncRead>>),

    #[cfg(not(target_family = "wasm"))]
    Response(Pin<Box<dyn AsyncRead + Send + Sync>>),
}

lazy_static! {
    /// A map from URLs to cached data. This assumes a url gives us the same data on each request (at least during
    /// the time this process is running).
    /// We already rely on this assumption in several places so this is okay.
    static ref FILE_INFO_CACHE: DashMap<String, CachedData> = DashMap::new();
}

struct CachedData {
    file_len: u64,
}

impl HTTPFile {
    pub async fn new(client: reqwest::Client, info: FileInfo) -> Result<HTTPFile> {
        // Check the cache
        let file_len = match FILE_INFO_CACHE.get(&info.url) {
            Some(cached_data) => cached_data.file_len,
            None => {
                // TODO: maybe lazily fetch this
                // TODO: include the URL in the error messages below
                let res = client.head(&info.url).send().await?;
                let file_len = res
                    .headers()
                    .get(reqwest::header::CONTENT_LENGTH)
                    .ok_or(CartonError::Other(
                        "Tried to fetch a URL that didn't have a content length",
                    ))?
                    .to_str()
                    .map_err(|_| {
                        CartonError::Other("Tried to fetch a URL with an invalid content length.")
                    })?
                    .parse()
                    .map_err(|_| {
                        CartonError::Other("Tried to fetch a URL with an invalid content length.")
                    })?;

                FILE_INFO_CACHE.insert(info.url.clone(), CachedData { file_len });

                file_len
            }
        };

        Ok(Self {
            client,
            info,
            file_len,
            seek_pos: 0,
            state: RequestState::None,
        })
    }
}

impl AsyncRead for HTTPFile {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        loop {
            match &mut self.state {
                RequestState::None => {
                    // We don't have a request in flight yet. Create one
                    let url = self.info.url.clone();

                    // reqwest::Client is just an Arc internally so it's fairly cheap for us to clone
                    let client = self.client.clone();
                    let range_start = self.seek_pos;

                    // We're already at the end of the file so let the reader know we're done
                    if range_start == self.file_len {
                        return Poll::Ready(Ok(()));
                    }

                    self.state = RequestState::Request(Box::pin(async move {
                        fetch(&client, &url, range_start).await
                    }));
                }
                RequestState::Request(v) => match v.as_mut().poll(cx) {
                    Poll::Ready(res) => self.state = RequestState::Response(Box::pin(res)),
                    Poll::Pending => return Poll::Pending,
                },
                RequestState::Response(res) => return res.as_mut().poll_read(cx, buf),
            }
        }
    }
}

#[cfg(not(target_family = "wasm"))]
type FetchReturnType = Box<dyn AsyncRead + Unpin + Send + Sync>;

#[cfg(target_family = "wasm")]
type FetchReturnType = Box<dyn AsyncRead + Unpin>;

async fn fetch(client: &reqwest::Client, url: &str, range_start: u64) -> FetchReturnType {
    log::trace!("Starting fetch: {url}");
    let res = client
        .get(url)
        .header(reqwest::header::RANGE, format!("bytes={range_start}-"))
        .send()
        .await
        .unwrap();

    if !res.status().is_success() {
        // TODO: return an error instead of panic
        panic!("Error fetching URL {}: {}", url, res.status());
    }

    // Convert from a stream into futures::io::AsyncRead
    let stream = res
        .bytes_stream()
        .map_err(|e| futures::io::Error::new(futures::io::ErrorKind::Other, e))
        .into_async_read();

    // To tokio::io::AsyncRead
    let stream = stream.compat();

    Box::new(stream)
}

#[cfg_attr(target_family = "wasm", async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait)]
impl ReadableFile for HTTPFile {
    async fn metadata(&self) -> std::io::Result<Metadata> {
        let accessed = Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "HttpFS does not support `accessed`",
        ));

        let created = Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "HttpFS does not support `created`",
        ));

        let modified = Err(std::io::Error::new(
            std::io::ErrorKind::Other,
            "HttpFS does not support `modified`",
        ));

        let file_type = FileType::new(false, true, false);
        let len = self.file_len;
        let permissions = Permissions::new(true);

        Ok(Metadata::new(
            accessed,
            created,
            modified,
            file_type,
            len,
            permissions,
        ))
    }

    async fn try_clone(&self) -> std::io::Result<Self> {
        // Need to share seek pos, read pos, etc
        todo!()
    }
}

pub(crate) struct HttpFS {
    /// Map from path to FileInfo
    files: HashMap<PathBuf, FileInfo>,

    client: reqwest::Client,
}

#[derive(Clone)]
pub(crate) struct FileInfo {
    pub url: String,

    // TODO: actually verify that the sha256 matches the expected value
    pub _sha256: String,
}

impl HasFileType for HttpFS {
    type FileType = HTTPFile;
}

impl HttpFS {
    pub fn new(client: reqwest::Client, files: HashMap<PathBuf, FileInfo>) -> Self {
        Self { files, client }
    }
}

#[cfg_attr(target_family = "wasm", async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait)]
impl ReadableFileSystem for HttpFS {
    // Open a file
    async fn open(&self, path: impl PathType) -> std::io::Result<Self::FileType>
    where
        Self::FileType: ReadableFile,
    {
        let p = path.as_ref();
        match self.files.get(p) {
            Some(info) => Ok(HTTPFile::new(self.client.clone(), info.clone())
                .await
                .unwrap()),
            None => Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "File not found",
            )),
        }
    }

    // These are almost identical to tokio::fs::...
    async fn canonicalize(&self, path: impl PathType) -> std::io::Result<PathBuf> {
        // Normalize the path
        let normalized = path_clean::clean(path.as_ref().as_str()).into();

        // Make sure it exists
        match self.files.get(&normalized) {
            Some(_) => Ok(normalized),
            None => Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "File not found",
            )),
        }
    }

    async fn metadata(&self, path: impl PathType) -> std::io::Result<Metadata> {
        self.open(path).await?.metadata().await
    }

    async fn read(&self, path: impl PathType) -> std::io::Result<Vec<u8>> {
        let p = path.as_ref();
        match self.files.get(p) {
            Some(info) => Ok(self
                .client
                .get(&info.url)
                .send()
                .await
                .unwrap()
                .bytes()
                .await
                .unwrap()
                .to_vec()),
            None => Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "File not found",
            )),
        }
    }

    type ReadDirPollerType = HttpReadDirPoller;

    async fn read_dir(
        &self,
        path: impl PathType,
    ) -> std::io::Result<ReadDir<Self::ReadDirPollerType, Self>> {
        let p = path.as_ref();
        let poller = HttpReadDirPoller {
            files: self
                .files
                .iter()
                .filter_map(|(k, _)| {
                    if k.starts_with(p) {
                        Some(k.clone())
                    } else {
                        None
                    }
                })
                .collect(),
        };

        Ok(ReadDir::new(poller, self))
    }

    async fn read_link(&self, _path: impl PathType) -> std::io::Result<PathBuf> {
        Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "File not a symlink",
        ))
    }

    async fn read_to_string(&self, path: impl PathType) -> std::io::Result<String> {
        let p = path.as_ref();
        match self.files.get(p) {
            Some(info) => Ok(self
                .client
                .get(&info.url)
                .send()
                .await
                .unwrap()
                .text()
                .await
                .unwrap()),
            None => Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "File not found",
            )),
        }
    }

    async fn symlink_metadata(&self, path: impl PathType) -> std::io::Result<Metadata> {
        // We don't support symlinks so these are the same
        self.metadata(path).await
    }
}

pub(crate) struct HttpReadDirPoller {
    files: VecDeque<PathBuf>,
}

impl<F> ReadDirPoller<F> for HttpReadDirPoller
where
    F: ReadableFileSystem,
    F::FileType: ReadableFile,
{
    fn poll_next_entry<'a>(
        &mut self,
        _cx: &mut std::task::Context<'_>,
        fs: &'a F,
    ) -> Poll<std::io::Result<Option<lunchbox::types::DirEntry<'a, F>>>> {
        std::task::Poll::Ready(Ok(self
            .files
            .pop_front()
            .map(|v| DirEntry::new(fs, v.file_name().unwrap().to_owned(), v))))
    }
}
