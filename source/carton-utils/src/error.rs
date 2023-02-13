use thiserror::Error;

pub type Result<T> = std::result::Result<T, DownloadError>;

#[non_exhaustive]
#[derive(Debug, Error)]
pub enum DownloadError {
    #[error("{0}")]
    FetchError(#[from] reqwest::Error),

    #[error("Sha256 Mismatch. Expected {expected}, but got {actual}")]
    Sha256Mismatch { actual: String, expected: String },

    #[error("Error: {0}")]
    Other(&'static str),
}
