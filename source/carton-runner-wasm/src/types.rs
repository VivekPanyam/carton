use color_eyre::{Report, Result};
use color_eyre::eyre::eyre;
use carton_runner_interface::types::{
    Tensor as CartonTensor,
	TensorStorage as CartonStorage,
};

use crate::component::{
	Tensor as WasmTensor,
	TensorNumeric,
	TensorString,
	Dtype
};

impl Into<CartonTensor> for WasmTensor {
	fn into(self) -> CartonTensor {
		todo!()
	}
}

impl TryFrom<CartonTensor> for WasmTensor {
	type Error = Report;

	fn try_from(value: CartonTensor) -> Result<Self> {
		Ok(match value {
			CartonTensor::Float(t) => WasmTensor::Numeric(t.into()),
			CartonTensor::Double(t) => WasmTensor::Numeric(t.into()),
			CartonTensor::I8(t) => WasmTensor::Numeric(t.into()),
			CartonTensor::I16(t) => WasmTensor::Numeric(t.into()),
			CartonTensor::I32(t) => WasmTensor::Numeric(t.into()),
			CartonTensor::I64(t) => WasmTensor::Numeric(t.into()),
			CartonTensor::U8(t) => WasmTensor::Numeric(t.into()),
			CartonTensor::U16(t) => WasmTensor::Numeric(t.into()),
			CartonTensor::U32(t) => WasmTensor::Numeric(t.into()),
			CartonTensor::U64(t) => WasmTensor::Numeric(t.into()),
			CartonTensor::String(t) => WasmTensor::String(t.into()),
			CartonTensor::NestedTensor(_) => return Err(eyre!("Nested tensors are not supported")),
		})
	}
}

impl<T: DTypeOf> From<CartonStorage<T>> for TensorNumeric {
	fn from(value: CartonStorage<T>) -> Self {
		let (buffer, shape) = unsafe { value.into_bytes() };
		TensorNumeric {
			buffer,
			dtype: T::dtype(),
			shape,
		}
	}
}

impl From<CartonStorage<String>> for TensorString {
	fn from(value: CartonStorage<String>) -> Self {
		let (buffer, shape) = value.into_vec();
		TensorString {
			buffer,
			shape,
		}
	}
}

trait DTypeOf {
	fn dtype() -> Dtype;
}

macro_rules! for_each_numeric {
    ($macro_name:ident) => {
        $macro_name!(f32, Dtype::F32);
        $macro_name!(f64, Dtype::F64);
        $macro_name!(i32, Dtype::I32);
        $macro_name!(i64, Dtype::I64);
        $macro_name!(u8, Dtype::Ui8);
        $macro_name!(u16, Dtype::Ui16);
        $macro_name!(u32, Dtype::Ui32);
        $macro_name!(u64, Dtype::Ui64);
    };
}

macro_rules! implement_dtypeof {
    ($type:ty, $enum_variant:expr) => {
        impl DTypeOf for $type {
            fn dtype() -> Dtype {
                $enum_variant
            }
        }
    };
}

for_each_numeric!(implement_dtypeof);