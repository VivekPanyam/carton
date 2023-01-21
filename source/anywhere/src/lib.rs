use lunchbox::types::{HasFileType, MaybeSend, MaybeSync};
pub use lunchbox::{ReadableFileSystem, WritableFileSystem};
use rpc::ServerBuilder;

mod file_ops;
pub mod path;
pub mod rpc;
mod serialize;
pub mod transport;
pub mod types;

pub trait Servable<'a, const WRITABLE: bool, const SEEKABLE: bool>: HasFileType + Sized
where
    Self::FileType: MaybeSend + MaybeSync,
{
    // Every server must be at least readable so we set that to true
    fn build_server(&'a self) -> ServerBuilder<'a, Self, true /* READABLE */, WRITABLE, SEEKABLE> {
        ServerBuilder::new(&self)
    }
}

impl<T: HasFileType + Sized, const WRITABLE: bool, const SEEKABLE: bool>
    Servable<'_, WRITABLE, SEEKABLE> for T
where
    Self::FileType: MaybeSend + MaybeSync,
{
}
