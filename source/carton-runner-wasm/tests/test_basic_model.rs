use std::collections::HashMap;
use std::fs;
use std::fs::File;
use carton_runner_wasm::{OutputMetadata, DType};
use serde_json::to_writer;

#[tokio::test]
fn test_pack() {
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