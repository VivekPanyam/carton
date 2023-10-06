wit_bindgen::generate!({
	world: "model",
	path: "./wit/lib.wit",
	exports: {
		world: Model
	}
});

pub use carton_wasm::lib::types::*;

mod candle;

struct Model;
