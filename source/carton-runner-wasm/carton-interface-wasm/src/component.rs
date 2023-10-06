wit_bindgen::generate!({
	world: "model",
	exports: {
		world: Model
	}
});

pub use carton_wasm::lib::types::*;