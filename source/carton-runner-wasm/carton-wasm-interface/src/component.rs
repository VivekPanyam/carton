pub use carton_wasm::lib::types::{Dtype, TensorNumeric, TensorString};

wit_bindgen::generate!({
    world: "model",
    path: "../wit",
    exports: {
        world: Model
    }
});

pub struct Model;