wit_bindgen::generate!({
	world: "infer",
	exports: {
		world: Model
	}
});

struct Model;

impl Guest for Model {
	fn infer() {
		todo!()
	}
}