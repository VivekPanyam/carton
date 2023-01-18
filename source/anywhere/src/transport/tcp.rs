use tokio::net::{TcpListener, TcpStream, ToSocketAddrs};

use crate::{
    types::AnywhereFS,
    wrapper::{MaybeSeekable, MaybeWritable, Readable},
};

use super::{framed, Transport};

pub async fn connect<A: ToSocketAddrs, const WRITABLE: bool, const SEEKABLE: bool>(
    addr: A,
) -> std::io::Result<AnywhereFS<WRITABLE, SEEKABLE>> {
    // Connect
    let stream = TcpStream::connect(addr).await?;
    let (recv, send) = stream.into_split();

    framed::connect(send, recv).await
}

/// Serves a filesystem over TCP
/// Note: this only works for exactly one client
pub(crate) async fn serve_fs<A: ToSocketAddrs, T>(
    fs: crate::rpc::AnywhereRPCServer<T>,
    addr: A,
) -> std::io::Result<()>
where
    T: Readable + MaybeWritable + MaybeSeekable,
{
    let listener = TcpListener::bind(addr).await?;
    let (stream, _) = listener.accept().await?;

    let (recv, send) = stream.into_split();

    framed::serve_fs(fs, send, recv).await;

    Ok(())
}

pub struct TcpTransport {}
impl Transport for TcpTransport {
    type Ret<T> = TcpTransportServer<T>;

    fn new<T>(inner: crate::rpc::AnywhereRPCServer<T>) -> Self::Ret<T> {
        TcpTransportServer { inner }
    }
}

pub struct TcpTransportServer<T> {
    inner: crate::rpc::AnywhereRPCServer<T>,
}

impl<T: Readable + MaybeWritable + MaybeSeekable> TcpTransportServer<T> {
    /// Serves a filesystem over TCP
    /// Note: this only works for exactly one client
    pub async fn serve<A: ToSocketAddrs>(self, addr: A) -> std::io::Result<()> {
        serve_fs(self.inner, addr).await
    }
}
