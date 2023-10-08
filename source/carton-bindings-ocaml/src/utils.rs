// Stolen from carton-bindings-c/src/utils.rs

use std::sync::OnceLock;

use tokio::runtime::Runtime;

/// A utility to lazily start a tokio runtime
pub(crate) fn runtime() -> &'static Runtime {
    static CELL: OnceLock<Runtime> = OnceLock::new();
    CELL.get_or_init(|| Runtime::new().unwrap())
}
