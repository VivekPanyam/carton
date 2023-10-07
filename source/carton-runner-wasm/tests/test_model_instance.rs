use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

use wasmtime::{Config, Engine};
use carton_runner_interface::types::{Tensor, TensorStorage};

use carton_runner_wasm::WASMModelInstance;

#[test]
fn test_model_instance() {
	color_eyre::install().unwrap();
	let mut file = File::open("tests/test_model/model.wasm").unwrap();
	let mut bytes = Vec::new();
	file.read_to_end(&mut bytes).unwrap();
	let mut config = Config::new();
	config.wasm_component_model(true);
	let engine = Engine::new(&config).unwrap();
	let mut model = WASMModelInstance::from_bytes(&engine, &bytes).unwrap();
	let mut s = TensorStorage::<f32>::new(vec![10u64]);
	s.view_mut().as_slice_mut().unwrap().clone_from_slice(&vec![1f32; 10]);
	let mut dummy = HashMap::new();
	dummy.insert("in1".to_string(), Tensor::Float(s));
	let out = model.infer(dummy).unwrap();
	let s = match out.get("out1") {
		Some(Tensor::Float(s)) => s,
		_ => panic!("Wrong type")
	};
	assert_eq!(
		s.view().as_slice().unwrap(),
		&vec![2f32; 10]
	)
}