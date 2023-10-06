use std::collections::HashMap;
use std::fs;
use std::fs::File;
use wasmtime::{Engine, Instance, Module, Store};
use carton_runner_wasm::{OutputMetadata, DType};
use serde_json::to_writer;


#[test]
fn test_mem_import() {
	let engine = Engine::default();
	let module = Module::from_file(&engine, "tests/model/model.wasm").unwrap();
	let mut store = Store::new(&engine, ());
	let instance = Instance::new(&mut store, &module, &[]).unwrap();
	let mem = instance.get_memory(&mut store, "memory").unwrap();
	let in1 = instance.get_global(&mut store, "INPUT1").unwrap();
	let in2 = instance.get_global(&mut store, "INPUT2").unwrap();
	let out = instance.get_global(&mut store, "OUTPUT").unwrap();
	let mut buf = [1u8; 80];
	let offset = in1.get(&mut store).i32().unwrap() as usize;
	mem.read(&mut store, offset, &mut buf).unwrap();
	assert_eq!(buf, [0u8; 80]);
}

#[tokio::test]
async fn test_pack() {
	let temp_dir = tempfile::tempdir().unwrap();

	let omd_file = File::create(temp_dir.path().join("output_md.json")).unwrap();
	let mut output = HashMap::new();
	output.insert("OUTPUT".to_string(), OutputMetadata {
		dtype: DType::F32,
		shape: vec![20],
	});
	to_writer(omd_file, &output).unwrap();

	fs::copy(
		"tests/basic_model/target/wasm32-unknown-unknown/release/basic_model.wasm",
		temp_dir.path().join("model.wasm")
	).unwrap();
}