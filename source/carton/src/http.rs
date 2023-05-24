use crate::error::{CartonError, Result};
use std::{pin::Pin, task::Poll};
use tokio::io::{AsyncRead, AsyncSeek};

/// HTTPFile implements [`AsyncRead`] and [`AsyncSeek`] on top of HTTP requests.
///
/// Given a URL, it makes range requests to fulfill read and seek requests.
///
/// Note: HTTPFile does not implement caching so small or repeated reads may not be efficient.
pub struct HTTPFile {
    client: reqwest::Client,
    url: String,
    file_len: u64,

    // The current seek position
    seek_pos: u64,

    /// The current request we're waiting on (if any)
    #[cfg(target_family = "wasm")]
    curr_request: Option<Pin<Box<dyn std::future::Future<Output = bytes::Bytes>>>>,

    #[cfg(not(target_family = "wasm"))]
    curr_request: Option<Pin<Box<dyn std::future::Future<Output = bytes::Bytes> + Send + Sync>>>,
}

impl HTTPFile {
    pub async fn new(client: reqwest::Client, url: String) -> Result<HTTPFile> {
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

        Ok(Self {
            client,
            url,
            file_len,
            seek_pos: 0,
            curr_request: None,
        })
    }
}

impl AsyncSeek for HTTPFile {
    /// Computes and stores the offset
    fn start_seek(mut self: Pin<&mut Self>, position: std::io::SeekFrom) -> std::io::Result<()> {
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
        if self.curr_request.is_none() {
            // We don't have a request in flight yet. Create one
            let url = self.url.clone();

            // reqwest::Client is just an Arc internally so it's fairly cheap for us to clone
            let client = self.client.clone();
            let range_start = self.seek_pos;
            let num_bytes = buf.remaining() as u64;

            if range_start == self.file_len {
                return Poll::Ready(Ok(()));
            }

            self.curr_request = Some(Box::pin(async move {
                fetch_range(&client, &url, range_start, num_bytes).await
            }));
        }

        match self.curr_request.as_mut().unwrap().as_mut().poll(cx) {
            Poll::Ready(res) => {
                // Update the seek pos
                self.seek_pos += res.len() as u64;

                // Add the data
                buf.put_slice(&res);

                // Reset the current request
                self.curr_request = None;

                Poll::Ready(Ok(()))
            }
            Poll::Pending => Poll::Pending,
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
