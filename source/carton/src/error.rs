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

use thiserror::Error;

pub type Result<T> = std::result::Result<T, CartonError>;

#[non_exhaustive]
#[derive(Debug, Error)]
pub enum CartonError {
    #[error("Filesystem '{0}' not supported on current platform")]
    UnsupportedFileSystem(&'static str),

    #[error("Invalid format for device: '{0}'. Expected `cpu`, a device index, or a UUID starting with GPU- or MIG-GPU-")]
    InvalidDeviceFormat(String),

    #[error("Got an unknown datatype: {0}")]
    UnknownDataType(String),

    #[error("Internal error: {0}. Please file a GitHub issue with repro steps if you can.")]
    UnexpectedInternalError(&'static str),

    #[error("{0}")]
    FetchError(#[from] reqwest::Error),

    #[error("IO error: {0}")]
    IOError(#[from] std::io::Error),

    #[error("Error parsing carton metadata: {0}")]
    ConfigParsingError(#[from] toml::de::Error),

    #[error("Runner reported error: {0}")]
    ErrorFromRunner(String),

    #[error("Error while parsing version: {0}")]
    SemverParseError(#[from] semver::Error),

    #[error("Error: {0}")]
    Other(&'static str),
}
