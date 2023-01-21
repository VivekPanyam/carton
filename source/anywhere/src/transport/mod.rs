use crate::rpc::AnywhereRPCServer;

pub mod framed;
pub mod serde;

#[cfg(feature = "tcp")]
pub mod tcp;

pub trait Transport {
    /// By convention, T should have an async `serve` method that consumes self.
    type Ret<A, B, C, D>;

    fn new<A, B, C, D>(inner: AnywhereRPCServer<A, B, C, D>) -> Self::Ret<A, B, C, D>;
}
