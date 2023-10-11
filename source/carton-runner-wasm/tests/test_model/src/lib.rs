use std::collections::HashMap;
use std::ops::Add;
use std::panic::panic_any;
use std::slice::from_raw_parts;

use carton_wasm::lib::types::{Dtype, TensorNumeric};

wit_bindgen::generate!({
    world: "model",
    path: "../../wit",
    exports: {
        world: Model
    }
});

struct Model;

fn bytes_to_vec<T: Clone>(b: &mut [u8]) -> Vec<T> {
	assert_eq!(b.len() % std::mem::size_of::<T>(), 0, "Invalid byte length");
	let len = b.len() / std::mem::size_of::<T>();
	let ptr = b.as_mut_ptr() as *mut T;
	unsafe { from_raw_parts(ptr, len).to_vec() }
}

fn vec_to_bytes<T: Clone>(t: &mut [T]) -> Vec<u8> {
	let len = t.len() * std::mem::size_of::<T>();
	let ptr = t.as_mut_ptr() as *mut u8;
	unsafe { from_raw_parts(ptr, len).to_vec() }
}

fn to_f32(t: &mut Tensor) -> Vec<f32> {
	let n = match t {
		Tensor::Numeric(t) => t,
		Tensor::String(_) => panic_any("Invalid tensor type")
	};
	bytes_to_vec::<f32>(&mut n.buffer)
}

impl Guest for Model {
	fn infer(in_: Vec<(String, Tensor)>) -> Vec<(String, Tensor)> {
		let mut inputs: HashMap<String, Tensor> = in_.into_iter().collect();
		let out = to_f32(inputs.get_mut("in1").unwrap());
		let len = out.len().clone() as u64;
		let mut out = out.into_iter().map(|x| x.add(1f32)).collect::<Vec<f32>>();
		let t = Tensor::Numeric(
			TensorNumeric {
				buffer: vec_to_bytes::<f32>(&mut out),
				dtype: Dtype::Float,
				shape: vec![len],
			}
		);
		vec![("out1".parse().unwrap(), t)]
	}
}
