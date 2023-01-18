use lunchbox::types::HasFileType;
pub use lunchbox::{ReadableFileSystem, WritableFileSystem};
use rpc::ServerBuilder;

pub mod path;
pub mod rpc;
mod serialize;
pub mod transport;
pub mod types;
mod file_ops;


pub trait Servable<'a, const WRITABLE: bool, const SEEKABLE: bool> : HasFileType + Sized where Self::FileType: Send + Sync
{
    // Every server must be at least readable so we set that to true
    fn build_server(
        &'a self,
    ) -> ServerBuilder<'a, Self, true /* READABLE */, WRITABLE, SEEKABLE> {
        ServerBuilder::new(&self)
    }
}

impl<T: HasFileType + Sized, const WRITABLE: bool, const SEEKABLE: bool> Servable<'_, WRITABLE, SEEKABLE> for T where Self::FileType: Send + Sync {}
