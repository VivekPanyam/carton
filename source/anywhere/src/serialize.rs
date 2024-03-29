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

//! This module implements serializable wrappers for some types that are not serializable.

use std::{io::SeekFrom, time::SystemTime};

use lunchbox::types::{Metadata, Permissions};
use serde::{Deserialize, Serialize};

// For std::io::ErrorKind
macro_rules! impl_from {
    (
        $(# $attr:tt)*
        enum $enum_name:ident {
            $( $item:ident , )*
        }
    ) => {
        // Declare the enum
        $(# $attr )*
        enum $enum_name {
            $( $item , )*
        }

        // std::io::ErrorKind -> ErrorKind
        impl From<std::io::ErrorKind> for ErrorKind {
            fn from(value: std::io::ErrorKind) -> Self {
                match value {
                    $(
                        std::io::ErrorKind::$item => Self::$item,
                    )*
                    _ => Self::Other,
                }
            }
        }

        // ErrorKind -> std::io::ErrorKind
        impl From<ErrorKind> for std::io::ErrorKind {
            fn from(value: ErrorKind) -> Self {
                match value {
                    $(
                        ErrorKind::$item => std::io::ErrorKind::$item,
                    )*
                }
            }
        }
    };
}

impl_from! {
// A subset of std::io::ErrorKind
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
enum ErrorKind {
    // The commented out values are unstable rust features
    NotFound,
    PermissionDenied,
    ConnectionRefused,
    ConnectionReset,
    // HostUnreachable,
    // NetworkUnreachable,
    ConnectionAborted,
    NotConnected,
    AddrInUse,
    AddrNotAvailable,
    // NetworkDown,
    BrokenPipe,
    AlreadyExists,
    WouldBlock,
    // NotADirectory,
    // IsADirectory,
    // DirectoryNotEmpty,
    // ReadOnlyFilesystem,
    // FilesystemLoop,
    // StaleNetworkFileHandle,
    InvalidInput,
    InvalidData,
    TimedOut,
    WriteZero,
    // StorageFull,
    // NotSeekable,
    // FilesystemQuotaExceeded,
    // FileTooLarge,
    // ResourceBusy,
    // ExecutableFileBusy,
    // Deadlock,
    // CrossesDevices,
    // TooManyLinks,
    // InvalidFilename,
    // ArgumentListTooLong,
    Interrupted,
    Unsupported,
    UnexpectedEof,
    OutOfMemory,
    Other,
}
}

// For std::io::Error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoError {
    kind: ErrorKind,
    msg: String,
}

impl From<std::io::Error> for IoError {
    fn from(value: std::io::Error) -> Self {
        IoError {
            kind: value.kind().into(),
            msg: value.to_string(),
        }
    }
}

impl From<IoError> for std::io::Error {
    fn from(value: IoError) -> Self {
        std::io::Error::new(value.kind.into(), value.msg)
    }
}

// For std::io::SeekFrom
#[derive(Serialize, Deserialize)]
#[serde(remote = "SeekFrom")]
pub enum SeekFromDef {
    Start(u64),
    End(i64),
    Current(i64),
}

// For lunchbox::types::Metadata
fn get_accessed(item: &Metadata) -> std::result::Result<SystemTime, IoError> {
    item.accessed().map_err(|e| e.into())
}

fn get_created(item: &Metadata) -> std::result::Result<SystemTime, IoError> {
    item.created().map_err(|e| e.into())
}

fn get_modified(item: &Metadata) -> std::result::Result<SystemTime, IoError> {
    item.modified().map_err(|e| e.into())
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(remote = "Metadata")]
pub struct SerializableMetadata {
    #[serde(getter = "get_accessed")]
    accessed: std::result::Result<SystemTime, IoError>,

    #[serde(getter = "get_created")]
    created: std::result::Result<SystemTime, IoError>,

    #[serde(getter = "get_modified")]
    modified: std::result::Result<SystemTime, IoError>,

    #[serde(getter = "Metadata::file_type")]
    file_type: lunchbox::types::FileType,

    #[serde(getter = "Metadata::len")]
    len: u64,

    #[serde(getter = "Metadata::permissions")]
    permissions: Permissions,
}

impl From<SerializableMetadata> for Metadata {
    fn from(value: SerializableMetadata) -> Self {
        Metadata::new(
            value.accessed.map_err(|e| e.into()),
            value.created.map_err(|e| e.into()),
            value.modified.map_err(|e| e.into()),
            value.file_type,
            value.len,
            value.permissions,
        )
    }
}

fn path_to_string(path: &lunchbox::path::PathBuf) -> String {
    path.as_str().to_owned()
}

// the RelativePath types use `deserialize_any` which doesn't work with bincode
// An impl for RelativePathBuf
#[derive(Debug, Serialize, Deserialize)]
#[serde(remote = "lunchbox::path::PathBuf")]
pub struct SerializableRelativePathBuf {
    #[serde(getter = "path_to_string")]
    inner: String,
}

impl From<SerializableRelativePathBuf> for lunchbox::path::PathBuf {
    fn from(value: SerializableRelativePathBuf) -> Self {
        value.inner.into()
    }
}
