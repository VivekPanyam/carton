//! You may be wondering "why not use protobuf/gRPC/tarpc/something else?"
//! I initially tried both tarpc and proto/grpc.
//! For proto, I either had to write codegen, macros for codegen, or manually write out the protos. And then I'd
//! have to write macros to actually hook the client and server up.
//! For tarpc, the messiness of `wrapper` and interactions with async_trait made things pretty complex.
//! The main macro I had to write for tarpc was pretty similar to the one below.
//!
//! By making that macro a bit larger, I could entirely remove tarpc as a dependency.
//!
//! The main usecase I had for `anywhere` already had an RPC system. I was already planning on using
//! pluggable transports with tarpc anyway so what I really needed were just
//! serde serializable request and response enums.
//!
//! You can use the transports in [`transports`] on their own or on top of an rpc framework (like tarpc)

use std::io::SeekFrom;
use std::pin::Pin;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

use dashmap::DashMap;
use lunchbox::path::PathBuf;
use lunchbox::types::HasFileType;
use lunchbox::types::MaybeSend;
use lunchbox::types::MaybeSync;
use lunchbox::types::Metadata;
use lunchbox::types::OpenOptions;
use lunchbox::types::Permissions;
use lunchbox::ReadableFileSystem;
use lunchbox::WritableFileSystem;
use lunchbox::types::WritableFile;
use tokio::io::AsyncSeek;
use tokio::sync::mpsc;
use tokio::sync::oneshot;

use serde::{Deserialize, Serialize};

use crate::file_ops::ReadableFileOps;
use crate::file_ops::SeekableFileOps;
use crate::file_ops::WritableFileOps;
use crate::serialize::{IoError, SeekFromDef, SerializableMetadata};
use crate::transport::Transport;
use crate::types;
use crate::types::FileHandle;
use crate::types::RPCPath;
use lunchbox::types::ReadableFile;

use paste::paste;

#[derive(Serialize, Deserialize, Debug)]
pub enum Capabilities {
    Read,
    ReadSeek,
    ReadWrite,
    ReadWriteSeek,
}

pub type MessageType = (AnywhereRPCRequest, oneshot::Sender<AnywhereRPCResponse>);

pub struct AnywhereRPCClient {
    outgoing: mpsc::Sender<MessageType>,
}

impl AnywhereRPCClient {
    pub fn new(tx: mpsc::Sender<MessageType>) -> AnywhereRPCClient {
        AnywhereRPCClient { outgoing: tx }
    }

    pub(crate) async fn try_to_fs<const W: bool, const S: bool>(
        self,
    ) -> std::io::Result<types::AnywhereFS<W, S>> {
        // TODO: ensure that W and S match what we get back from the server
        // match self.get_fs_type().await.unwrap() {
        //     Capabilities::Read => todo!(),
        //     Capabilities::ReadSeek => todo!(),
        //     Capabilities::ReadWrite => todo!(),
        //     Capabilities::ReadWriteSeek => todo!(),
        // }

        Ok(types::AnywhereFS {
            client: Arc::new(self),
        })
    }
}

// pub struct AnywhereRPCServer<T> {
//     inner: T,
// }

// impl<T> AnywhereRPCServer<T>
// {
//     pub fn new(fs: T) -> Self {
//         Self { inner: fs }
//     }

//     pub fn with_transport<U: Transport<WRITABLE, SEEKABLE>>(self) -> U::Ret<T> {
//         U::new(self)
//     }
// }


#[cfg(target_family = "wasm")]
pub type BoxFuture<'a, T> = Pin<Box<dyn std::future::Future<Output = T> + 'a>>;

#[cfg(not(target_family = "wasm"))]
pub type BoxFuture<'a, T> = Pin<Box<dyn std::future::Future<Output = T> + Send + 'a>>;

macro_rules! maybe_add_args {
    ($context:ident, with_server_context) => { $context };
}

// This automatically implements an RPC system for a set of async functions
macro_rules! autoimpl {
    (
        $(
            SECTION: $section_name:ident requires $( $required_traits:ident ),+ $(; filetype requires $( $filetype_required_traits:ident ),+ )? {
                $(
                    $(#[ $fn_attr:ident ])?
                    async fn $fn_name:ident ( $( $(# $attr:tt )? $arg_name:ident : $arg_type: ty),* ) -> std::io::Result<$(# $res_attr:tt )? $res_type:ty>;
                )*
            }
        )+
    ) => {
        paste! {
            // Specialization isn't stable yet so we have to do this somewhat hacky thing
            // For each section:
            $(
                // Create an "Allow" wrapper struct
                pub struct [< Allow $section_name >]<'a, T, const ALLOW: bool> {
                    pub(crate) inner: &'a T
                }

                // Create a "Maybe" trait
                pub trait [<Maybe $section_name>] <ContextType> {
                    $(
                        fn $fn_name <'a, 'c: 'a> ( &'a self, context: &'c ContextType,  $($arg_name: $arg_type),* ) -> BoxFuture<'a, std::io::Result<$res_type>>;
                    )*
                }

                // impl "Maybe" for "Allow" that fails when not allowed
                impl <'a, T, ContextType> [<Maybe $section_name>]<ContextType> for [< Allow $section_name >]<'a, T, false>
                {
                    $(
                        fn $fn_name <'b, 'c: 'b> ( &self, context: &ContextType, $($arg_name: $arg_type),* ) -> BoxFuture<std::io::Result<$res_type>> {
                            // TODO: return an error instead of panicking
                            panic!("Tried calling {} on a filesystem that does not support it", stringify!($fn_name));
                        }
                    )*
                }

                // impl "Maybe" for "Allow" when T meets the required traits and is allowed
                impl <'a, T: $( $required_traits + )+> [<Maybe $section_name>]<ServerContext<T>> for [< Allow $section_name >]<'a, T, true> where T::FileType: MaybeSend + MaybeSync $( + $( $filetype_required_traits + )+ )?
                {
                    $(
                        fn $fn_name <'b, 'c: 'b> ( &'b self, context: &'c ServerContext<T>,  $($arg_name: $arg_type),* ) -> BoxFuture<'b, std::io::Result<$res_type>> {
                            // Pass through
                            // This is kinda hacky
                            self.inner.$fn_name($(maybe_add_args!(context, $fn_attr), )? $( $arg_name ),*)
                        }
                    )*
                }
            )+

            // Request type
            #[derive(Serialize, Deserialize, Debug)]
            pub enum AnywhereRPCRequest {
                // For each section
                $(
                    // For each method
                    $(
                        [<$fn_name:camel>] {
                            $( $(# $attr )? $arg_name : $arg_type ),*
                        },
                    )*
                )+
            }

            // Response type
            #[derive(Serialize, Deserialize, Debug)]
            pub enum AnywhereRPCResponse {
                IoError(IoError),
                // For each section
                $(
                    // For each method
                    $(
                        [<$fn_name:camel>] {
                            $(# $res_attr )?
                            res: $res_type
                        },
                    )*
                )+
            }

            // Client
            // This implements methods for all the sections
            impl AnywhereRPCClient {
                // For each section
                $(
                    // For each method
                    $(
                        pub async fn $fn_name (&self, $($arg_name: $arg_type),* ) -> std::io::Result<$res_type> {
                            let req = AnywhereRPCRequest::[<$fn_name:camel>] { $( $arg_name ),* };
                            let (tx, rx) = oneshot::channel();

                            if self.outgoing.send((req, tx)).await.is_err() {
                                panic!("Error making RPC request");
                            }

                            match rx.await {
                                Ok(item) => {
                                    match item {
                                        AnywhereRPCResponse::[<$fn_name:camel>] { res } => Ok(res),
                                        AnywhereRPCResponse::IoError(e) => Err(e.into()),
                                        _ => panic!("Got unexpected type in RPC response"),
                                    }
                                },
                                Err(_) => panic!("Sender dropped without message")
                            }
                        }
                    )*
                )+
            }

            // The overall server struct that contains each of the `allows` above
            pub struct AnywhereRPCServer<ContextType, $( [<SomethingMaybe $section_name>] ),+> {
                context: ContextType,
                $(
                    [< $section_name:snake >]: [<SomethingMaybe $section_name>],
                )+
            }

            // Implement a server builder
            pub struct ServerBuilder<'a, T, $( const [<ALLOW_ $section_name:snake:upper>] : bool),+> {
                fs: &'a T
            }

            impl<'a, T: HasFileType, $( const [<ALLOW_ $section_name:snake:upper>] : bool),+> ServerBuilder<'a, T, $( [<ALLOW_ $section_name:snake:upper>] ),+> where T::FileType: MaybeSend + MaybeSync {
                pub(crate) fn new(fs: &'a T) -> Self {
                    Self { fs }
                }

                pub fn build(
                    &self
                ) -> AnywhereRPCServer<ServerContext<T>, $([<Allow $section_name>]<'a, T, [<ALLOW_ $section_name:snake:upper>]>),+>
                where
                    $([<Allow $section_name>]<'a, T, [<ALLOW_ $section_name:snake:upper>]> : [<Maybe $section_name>]<ServerContext<T>>),+

                {
                    AnywhereRPCServer {
                        context: ServerContext::<T>::new(),
                        $(
                            [< $section_name:snake >]: [<Allow $section_name>] { inner: self.fs},
                        )+
                    }
                }

                // These methods act as hints to the compiler so it can infer types
                $(
                    pub fn [< allow_ $section_name:snake>](&self) -> &Self
                    where
                        // This is horrible, but we can't use const generics in where clauses and
                        // writing separate impls requires spelling out all the ALLOW_*. This would
                        // effectively be a nested loop over section_name with correct placement of
                        // `true` or `false` in the struct generic parameters. That's complex to do
                        // in a macro so instead we do this
                        Assert<{ [< ALLOW_ $section_name:snake:upper >] }>: IsTrue,
                    {
                        self
                    }

                    pub fn [< disallow_ $section_name:snake>](&self) -> &Self
                    where
                        // This is horrible, but the alternatives aren't great. See
                        // the comment immediately above
                        Assert<{ [< ALLOW_ $section_name:snake:upper >] }>: IsFalse,
                    {
                        self
                    }
                )+

            }


            impl<ContextType, $( [<SomethingMaybe $section_name>] : [<Maybe $section_name>] <ContextType> ),+> AnywhereRPCServer<ContextType, $( [<SomethingMaybe $section_name>]),+> {
                pub fn into_transport<U: Transport>(self) -> U::Ret<ContextType, $( [<SomethingMaybe $section_name>]),+> {
                    U::new(self)
                }

                pub(crate) async fn handle_message(&self, req: AnywhereRPCRequest) -> AnywhereRPCResponse {
                    match req {
                        // For each section
                        $(
                            // For each method
                            $(
                                AnywhereRPCRequest::[<$fn_name:camel>] {
                                    $( $arg_name ),*
                                } => {
                                    let out = self.[< $section_name:snake >].$fn_name(&self.context, $( $arg_name ),*).await;

                                    // Store the item or the error
                                    match out {
                                        Ok(item) => AnywhereRPCResponse::[<$fn_name:camel>] {res: item},
                                        Err(e) => AnywhereRPCResponse::IoError(e.into()),
                                    }
                                },
                            )*
                        )+
                    }
                }
            }

        }
    };
}

pub struct ServerContext<T: HasFileType> where T::FileType: MaybeSend + MaybeSync {
    pub(crate) open_files: DashMap<FileHandle, T::FileType>,
    pub(crate) file_handle_counter: AtomicU64,
}

impl<T: HasFileType> ServerContext<T> where T::FileType: MaybeSend + MaybeSync {
    fn new() -> Self {
        Self {
            open_files: Default::default(),
            file_handle_counter: 0.into(),
        }
    }
}

autoimpl! {
    // The following methods need a readable filesystem
    SECTION: Read requires ReadableFileSystem, ReadableFileOps, Sync; filetype requires ReadableFile, Unpin
    {
        // // Get filesystem capabilities given a token
        // async fn get_fs_type() -> std::io::Result<Capabilities>;

        // File IO
        #[with_server_context]
        async fn read_bytes(handle: FileHandle, max_num_bytes: u64) -> std::io::Result<#[serde(with = "serde_bytes")] Vec<u8>>;

        // Read only file operations
        #[with_server_context]
        async fn file_metadata(handle: FileHandle) -> std::io::Result<#[serde(with = "SerializableMetadata")] Metadata>;

        #[with_server_context]
        async fn file_try_clone(handle: FileHandle) -> std::io::Result<FileHandle>;

        // Read only filesystem operations
        #[with_server_context]
        async fn open_file(path: RPCPath) -> std::io::Result<FileHandle>;
        async fn canonicalize(path: RPCPath) -> std::io::Result<PathBuf>;
        async fn metadata(path: RPCPath) -> std::io::Result<#[serde(with = "SerializableMetadata")] Metadata>;
        async fn read(path: RPCPath) -> std::io::Result<Vec<u8>>;
        // async fn read_dir(path: RPCPath) -> std::io::Result<ReadDir<Self::ReadDirPollerType, Self>>;
        async fn read_link(path: RPCPath) -> std::io::Result<PathBuf>;
        async fn read_to_string(path: RPCPath) -> std::io::Result<String>;
        async fn symlink_metadata(path: RPCPath) -> std::io::Result<#[serde(with = "SerializableMetadata")] Metadata>;
    }

    // The following methods need a writable filesystem
    SECTION: Write requires WritableFileSystem, WritableFileOps, Sync; filetype requires WritableFile, Unpin
    {
        // File IO
        #[with_server_context]
        async fn write_data(handle: FileHandle, #[serde(with = "serde_bytes")] buf: Vec<u8>) -> std::io::Result<usize>;

        #[with_server_context]
        async fn write_flush(handle: FileHandle) -> std::io::Result<()>;

        #[with_server_context]
        async fn write_shutdown(handle: FileHandle) -> std::io::Result<()>;

        // Write file operations
        #[with_server_context]
        async fn file_sync_all(handle: FileHandle) -> std::io::Result<()>;

        #[with_server_context]
        async fn file_sync_data(handle: FileHandle) -> std::io::Result<()>;

        #[with_server_context]
        async fn file_set_len(handle: FileHandle, size: u64) -> std::io::Result<()>;

        #[with_server_context]
        async fn file_set_permissions(handle: FileHandle, perm: Permissions) -> std::io::Result<()>;

        // Write filesystem operations
        #[with_server_context]
        async fn create_file(path: RPCPath) -> std::io::Result<FileHandle>;

        #[with_server_context]
        async fn open_file_with_opts(opts: OpenOptions, path: RPCPath) -> std::io::Result<FileHandle>;
        async fn copy(from: RPCPath, to: RPCPath) -> std::io::Result<u64>;
        async fn create_dir(path: RPCPath) -> std::io::Result<()>;
        async fn create_dir_all(path: RPCPath) -> std::io::Result<()>;
        async fn hard_link(src: RPCPath, dst: RPCPath) -> std::io::Result<()>;
        async fn remove_dir(path: RPCPath) -> std::io::Result<()>;
        async fn remove_dir_all(path: RPCPath) -> std::io::Result<()>;
        async fn remove_file(path: RPCPath) -> std::io::Result<()>;
        async fn rename(from: RPCPath, to: RPCPath) -> std::io::Result<()>;
        async fn set_permissions(path: RPCPath, perm: Permissions) -> std::io::Result<()>;
        async fn symlink(src: RPCPath, dst: RPCPath) -> std::io::Result<()>;
        async fn write(path: RPCPath, #[serde(with = "serde_bytes")] contents: Vec<u8>) -> std::io::Result<()>;
    }

    // // The following methods need a seekable filesystem
    SECTION: Seek requires SeekableFileOps, Sync; filetype requires AsyncSeek, Unpin
    {
        // File IO
        #[with_server_context]
        async fn seek(handle: FileHandle, #[serde(with = "SeekFromDef")] pos: SeekFrom) -> std::io::Result<u64>;
    }
}

pub struct Assert<const COND: bool> {}
pub trait IsTrue {}
impl IsTrue for Assert<true> {}

pub trait IsFalse {}
impl IsFalse for Assert<false> {}

#[cfg(test)]
mod tests {
    // use std::sync::Arc;

    // use tokio::sync::mpsc;

    // use super::{AnywhereRPCClient, AnywhereRPCServer};

    // #[tokio::test]
    // async fn test() {
    //     // Create a Readable and Writable FS by wrapping a lunchbox LocalFS
    //     let wrapped = FSWrapper::<_, true, false>::new(lunchbox::LocalFS::new().unwrap());

    //     // Create a channel to use as our transport
    //     let (tx, mut rx) = mpsc::channel(32);

    //     // Create a new AnywhereRPCClient
    //     let client = AnywhereRPCClient::new(tx);

    //     // Create a new AnywhereRPCServer
    //     let server = AnywhereRPCServer::new(Arc::new(wrapped));

    //     tokio::spawn(async move {
    //         // Handle the messages
    //         while let Some((req, res_sender)) = rx.recv().await {
    //             let _ = res_sender.send(server.handle_message(req).await);
    //         }
    //     });

    //     let out = client.read_to_string("/tmp/test.txt".into()).await.unwrap();
    //     println!("{}", out);
    // }
}
