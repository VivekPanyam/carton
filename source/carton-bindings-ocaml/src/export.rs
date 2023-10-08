use crate::*;

#[ocaml_rust::bridge]
mod ffi {
    type FFICarton = Custom<Carton>;
    type FFICartonError = Custom<CartonError>;

    #[derive(Clone, Debug)]
    pub enum Tensor {
        Float(Vec<f32>),
        Double(Vec<f64>),
        I8(Vec<i64>),
        I16(Vec<i64>),
        I32(Vec<i64>),
        I64(Vec<i64>),
        U8(Vec<i64>),
        U16(Vec<i64>),
        U32(Vec<i64>),
        U64(Vec<i64>),
        String(Vec<String>),
    }

    extern "Rust" {
        fn load(
            path: String,
            visible_device: Option<String>,
            override_runner_name: Option<String>,
            override_required_framework_version: Option<String>,
        ) -> Result<FFICarton, FFICartonError>;

        fn infer(model: FFICarton, tensors: Vec<(String, Tensor)>) -> Vec<(String, Tensor)>;
    }
}
