use carton_core::types::{GenericStorage, Tensor, TensorStorage, TypedStorage};

use crate::Tensor as SupportedTensorType;

macro_rules! from_tensor_int {
    ( $over:ident, $x:expr ) => {
        SupportedTensorType::$over(
            $x.view()
                .into_owned()
                .into_raw_vec()
                .into_iter()
                .map(|x| x as i64)
                .collect(),
        )
    };
}

macro_rules! to_tensor_int {
    ( $over:ident, $x:expr, $as:ident ) => {
        Tensor::<GenericStorage>::$over(
            ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[1]),
                $x.into_iter().map(|x| x as $as).collect(),
            )
            .unwrap(),
        )
    };
}

impl<T: TensorStorage> From<Tensor<T>> for SupportedTensorType {
    fn from(item: Tensor<T>) -> Self {
        match item {
            Tensor::Float(x) => SupportedTensorType::Float(x.view().into_owned().into_raw_vec()),
            Tensor::Double(x) => SupportedTensorType::Double(x.view().into_owned().into_raw_vec()),
            Tensor::I8(x) => from_tensor_int!(I8, x),
            Tensor::I16(x) => from_tensor_int!(I16, x),
            Tensor::I32(x) => from_tensor_int!(I32, x),
            Tensor::I64(x) => from_tensor_int!(I64, x),
            Tensor::U8(x) => from_tensor_int!(U8, x),
            Tensor::U16(x) => from_tensor_int!(U16, x),
            Tensor::U32(x) => from_tensor_int!(U32, x),
            Tensor::U64(x) => from_tensor_int!(U64, x),
            Tensor::String(x) => SupportedTensorType::String(x.view().into_owned().into_raw_vec()),
            Tensor::NestedTensor(_) => panic!("Nested tensor output not implemented yet"),
        }
    }
}

impl From<SupportedTensorType> for Tensor<GenericStorage> {
    fn from(item: SupportedTensorType) -> Self {
        match item {
            SupportedTensorType::Float(x) => to_tensor_int!(Float, x, f32),
            SupportedTensorType::Double(x) => to_tensor_int!(Double, x, f64),
            SupportedTensorType::String(x) => to_tensor_int!(String, x, String),
            SupportedTensorType::I8(x) => to_tensor_int!(I8, x, i8),
            SupportedTensorType::I16(x) => to_tensor_int!(I16, x, i16),
            SupportedTensorType::I32(x) => to_tensor_int!(I32, x, i32),
            SupportedTensorType::I64(x) => to_tensor_int!(I64, x, i64),
            SupportedTensorType::U8(x) => to_tensor_int!(U8, x, u8),
            SupportedTensorType::U16(x) => to_tensor_int!(U16, x, u16),
            SupportedTensorType::U32(x) => to_tensor_int!(U32, x, u32),
            SupportedTensorType::U64(x) => to_tensor_int!(U64, x, u64),
        }
    }
}
