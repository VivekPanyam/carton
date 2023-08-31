use std::sync::OnceLock;

pub fn init_logging() {
    // When the `console_error_panic_hook` feature is enabled, we can call the
    // `set_panic_hook` function at least once during initialization, and then
    // we will get better error messages if our code ever panics.
    //
    // For more details see
    // https://github.com/rustwasm/console_error_panic_hook#readme
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();

    // Init logging to trace
    // TODO: change this to Info for release builds
    static CELL: OnceLock<()> = OnceLock::new();
    CELL.get_or_init(|| console_log::init_with_level(log::Level::Trace).unwrap());
}
