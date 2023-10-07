#[macro_export]
macro_rules! generate_bindings {
    () => {
        wit_bindgen::generate!({
            world: "model",
            path: "../wit",
            exports: {
                world: Model
            }
        });
    };
}

mod component;