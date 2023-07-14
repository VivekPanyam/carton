use crate::error::{CartonError, Result};
use dashmap::DashMap;
use futures::TryStreamExt;
use lazy_static::lazy_static;
use std::{pin::Pin, sync::Arc, task::Poll};
use tokio::io::{AsyncRead, AsyncSeek};
use tokio_util::compat::FuturesAsyncReadCompatExt;

/// HTTPFile implements [`AsyncRead`] and [`AsyncSeek`] on top of HTTP requests.
///
/// Given a URL, it makes range requests to fulfill read and seek requests.
///
/// Note: HTTPFile does not implement general caching so small or repeated reads may not be efficient.
/// It does cache the last 64kb to make zip file loading faster.
pub struct HTTPFile {
    client: reqwest::Client,
    url: String,
    file_len: u64,

    // The current seek position
    seek_pos: u64,

    state: RequestState,

    cached_data: Arc<CachedData>,
}

enum RequestState {
    None,

    /// The current request we're waiting on (if any)
    #[cfg(target_family = "wasm")]
    Request(Pin<Box<dyn std::future::Future<Output = FetchReturnType>>>),

    #[cfg(not(target_family = "wasm"))]
    Request(Pin<Box<dyn std::future::Future<Output = FetchReturnType> + Send + Sync>>),

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
    static ref FILE_INFO_CACHE: DashMap<String, Arc<CachedData>> = DashMap::new();
}

struct CachedData {
    file_len: u64,

    // The zipfile implementation scans ~the last 66k bytes looking for
    // the EOCDR for Zip and Zip64. For non-zip64 files, this means it always scans the last ~66kb in 2kb chunks
    // We don't want to make 33 http requests just for that so we'll fetch and cache it
    file_end_data: Vec<u8>,
}

impl HTTPFile {
    pub async fn new(client: reqwest::Client, url: String) -> Result<HTTPFile> {
        let cached_data = match FILE_INFO_CACHE.get(&url) {
            Some(len) => len.clone(),
            None => {
                // TODO: because the registry site isn't public yet, we need to check for the `x-carton-dl-url` header
                // _before_ following redirects. This can be removed after the site is made public
                let res = client.head(&url).send().await?;

                // TODO: maybe lazily fetch this
                // TODO: include the URL in the error messages below
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

                const NUM_END_BYTES: u64 = 66 * 1024;

                let cached_data = Arc::new(CachedData {
                    file_len,
                    file_end_data: fetch_range(
                        &client,
                        &url,
                        file_len.saturating_sub(NUM_END_BYTES),
                        NUM_END_BYTES,
                    )
                    .await
                    .into(),
                });

                FILE_INFO_CACHE.insert(url.clone(), cached_data.clone());

                cached_data
            }
        };

        Ok(Self {
            client,
            url,
            file_len: cached_data.file_len,
            seek_pos: 0,
            state: RequestState::None,
            cached_data,
        })
    }
}

impl AsyncSeek for HTTPFile {
    /// Computes and stores the offset
    fn start_seek(mut self: Pin<&mut Self>, position: std::io::SeekFrom) -> std::io::Result<()> {
        // Clear the current request
        // TODO: only do this if the seek position is actually different
        self.state = RequestState::None;
        match position {
            std::io::SeekFrom::Start(newpos) => {
                self.seek_pos = newpos.min(self.file_len);
            }
            std::io::SeekFrom::End(offset) => {
                self.seek_pos = self
                    .file_len
                    .saturating_add_signed(offset)
                    .min(self.file_len);
            }
            std::io::SeekFrom::Current(offset) => {
                self.seek_pos = self
                    .seek_pos
                    .saturating_add_signed(offset)
                    .min(self.file_len);
            }
        }

        Ok(())
    }

    /// Returns the stored offset
    fn poll_complete(
        self: Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> Poll<std::io::Result<u64>> {
        Poll::Ready(Ok(self.seek_pos))
    }
}

impl AsyncRead for HTTPFile {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        let range_start = self.seek_pos;

        // If we're within our cached data
        let cache_start_pos = self
            .file_len
            .saturating_sub(self.cached_data.file_end_data.len() as _);
        if range_start >= cache_start_pos {
            // Offset into the cache
            let cache_offset = (range_start - cache_start_pos) as _;

            // Update the seek pos
            let num_bytes = self
                .file_len
                .saturating_sub(range_start)
                .min(buf.remaining() as _);
            self.seek_pos += num_bytes;

            // Add the data
            buf.put_slice(
                &self.cached_data.file_end_data[cache_offset..(cache_offset + num_bytes as usize)],
            );

            return Poll::Ready(Ok(()));
        }

        loop {
            match &mut self.state {
                RequestState::None => {
                    // We don't have a request in flight yet. Create one
                    let url = self.url.clone();

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
                RequestState::Response(res) => {
                    let num_bytes_orig = buf.remaining();
                    let out = res.as_mut().poll_read(cx, buf);
                    let num_bytes_end = buf.remaining();

                    // Update the seek pos
                    self.seek_pos += (num_bytes_orig - num_bytes_end) as u64;

                    return out;
                }
            }
        }
    }
}

/// Fetch a range of bytes from a URL via HTTP range requests
async fn fetch_range(
    client: &reqwest::Client,
    url: &str,
    range_start: u64,
    num_bytes: u64,
) -> bytes::Bytes {
    println!("Request: {} {}", range_start, num_bytes);
    let range_end = range_start + num_bytes - 1;
    let res = client
        .get(url)
        .header(
            reqwest::header::RANGE,
            format!("bytes={range_start}-{range_end}"),
        )
        .send()
        .await
        .unwrap();

    if !res.status().is_success() {
        // TODO: return an error instead of panic
        panic!("Error fetching URL {}: {}", url, res.status());
    }

    res.bytes().await.unwrap()
}

#[cfg(not(target_family = "wasm"))]
type FetchReturnType = Box<dyn AsyncRead + Unpin + Send + Sync>;

#[cfg(target_family = "wasm")]
type FetchReturnType = Box<dyn AsyncRead + Unpin>;

async fn fetch(client: &reqwest::Client, url: &str, range_start: u64) -> FetchReturnType {
    println!("Request: {}", range_start);
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
