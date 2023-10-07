use color_eyre::{Report, Result};
use color_eyre::eyre::{ensure, eyre};
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
		match self {
			WasmTensor::Numeric(t) => t.into(),
			WasmTensor::Str(t) => t.into(),
		}
	}
}

fn bytes_to_slice<T>(b: &[u8]) -> Result<&[T]> {
	ensure!(b.len() % std::mem::size_of::<T>() == 0, "Invalid byte length");
	Ok(unsafe {
		std::slice::from_raw_parts(
			b.as_ptr() as *const T,
			b.len() / std::mem::size_of::<T>(),
		)
	})
}

fn copy_to_storage<T: Clone + Default>(mut s: CartonStorage<T>, b: &[u8]) -> CartonStorage<T> {
	s.view_mut()
		.as_slice_mut()
		.unwrap()
		.clone_from_slice(bytes_to_slice(b).unwrap());
	s
}

impl Into<CartonTensor> for TensorNumeric {
	fn into(self) -> CartonTensor {
		match self.dtype {
			Dtype::F32 => copy_to_storage(
				CartonStorage::<f32>::new(self.shape),
				&self.buffer
			).into(),
			Dtype::F64 => copy_to_storage(
				CartonStorage::<f64>::new(self.shape),
				&self.buffer
			).into(),
			Dtype::I8 => copy_to_storage(
				CartonStorage::<i8>::new(self.shape),
				&self.buffer
			).into(),
			Dtype::I16 => copy_to_storage(
				CartonStorage::<i16>::new(self.shape),
				&self.buffer
			).into(),
			Dtype::I32 => copy_to_storage(
				CartonStorage::<i32>::new(self.shape),
				&self.buffer
			).into(),
			Dtype::I64 => copy_to_storage(
				CartonStorage::<i64>::new(self.shape),
				&self.buffer
			).into(),
			Dtype::Ui8 => copy_to_storage(
				CartonStorage::<u8>::new(self.shape),
				&self.buffer
			).into(),
			Dtype::Ui16 => copy_to_storage(
				CartonStorage::<u16>::new(self.shape),
				&self.buffer
			).into(),
			Dtype::Ui32 => copy_to_storage(
				CartonStorage::<u32>::new(self.shape),
				&self.buffer
			).into(),
			Dtype::Ui64 => copy_to_storage(
				CartonStorage::<u64>::new(self.shape),
				&self.buffer
			).into(),
		}
	}
}

impl Into<CartonTensor> for TensorString {
	fn into(self) -> CartonTensor {
		let mut t = CartonStorage::new(self.shape);
		t.view_mut().as_slice_mut().unwrap().clone_from_slice(&self.buffer);
		t.into()
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
			CartonTensor::String(t) => WasmTensor::Str(t.into()),
			CartonTensor::NestedTensor(_) => return Err(eyre!("Nested tensors are not supported")),
		})
	}
}

fn slice_to_bytes<T>(s: &[T]) -> &[u8] {
	unsafe {
		std::slice::from_raw_parts(
			s.as_ptr() as *const u8,
			s.len() * std::mem::size_of::<T>(),
		)
	}
}

impl<T: DTypeOf> From<CartonStorage<T>> for TensorNumeric {
	fn from(value: CartonStorage<T>) -> Self {
		let view = value.view();
		let shape = view.shape().iter().map(|&x| x as u64).collect();
		let buffer = view.as_slice().unwrap();
		TensorNumeric {
			buffer: slice_to_bytes(buffer).to_vec(),
			dtype: T::dtype(),
			shape,
		}
	}
}

impl From<CartonStorage<String>> for TensorString {
	fn from(value: CartonStorage<String>) -> Self {
		let view = value.view();
		let shape = view.shape().iter().map(|&x| x as u64).collect();
		let buffer = view.as_slice().unwrap().to_vec();
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
		$macro_name!(i8, Dtype::I8);
		$macro_name!(i16, Dtype::I16);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wasm_tensor_to_carton_tensor_f32() {
        let buffer = slice_to_bytes(&[1.0f32, 2.0f32, 3.0f32]);
        let tensor = WasmTensor::Numeric(TensorNumeric {
            buffer: buffer.to_vec(),
            dtype: Dtype::F32,
            shape: vec![3],
        });
        let carton_tensor: CartonTensor = tensor.into();
        if let CartonTensor::Float(storage) = carton_tensor {
            assert_eq!(
				storage.view().as_slice().unwrap(),
				&[1.0f32, 2.0f32, 3.0f32]
			);
        } else {
            return panic!("Expected CartonTensor::Float variant");
        }
    }

    #[test]
    fn carton_tensor_to_wasm_tensor_f32() {
        let storage = CartonStorage::<f32>::new(vec![3]);
        let carton_tensor = CartonTensor::Float(
			copy_to_storage(
				storage,
				slice_to_bytes(
					&[1.0f32, 2.0f32, 3.0f32]
				)
			)
		);
        let wasm_tensor = WasmTensor::try_from(carton_tensor).unwrap();
        if let WasmTensor::Numeric(tensor_numeric) = wasm_tensor {
            let data = bytes_to_slice::<f32>(&tensor_numeric.buffer).unwrap();
            assert_eq!(data, &[1.0f32, 2.0f32, 3.0f32]);
            assert_eq!(tensor_numeric.dtype, Dtype::F32);
        } else {
            return panic!("Expected WasmTensor::Numeric variant");
        }
    }

	#[test]
    fn wasm_tensor_to_carton_tensor_string() {
        let buffer = vec!["hello".to_string(), "world".to_string()];
        let tensor = WasmTensor::Str(TensorString {
            buffer: buffer.clone(),
            shape: vec![2],
        });
        let carton_tensor: CartonTensor = tensor.into();
        if let CartonTensor::String(storage) = carton_tensor {
            assert_eq!(storage.view().as_slice().unwrap(), &buffer);
        } else {
            return panic!("Expected CartonTensor::String variant");
        }
    }

    #[test]
    fn carton_tensor_to_wasm_tensor_string() {
        let buffer = vec!["hello".to_string(), "world".to_string()];
        let mut storage = CartonStorage::<String>::new(vec![2]);
        storage.view_mut().as_slice_mut().unwrap().clone_from_slice(&buffer);
		let carton_tensor = CartonTensor::String(storage);
        let wasm_tensor = WasmTensor::try_from(carton_tensor).unwrap();
        if let WasmTensor::Str(tensor_string) = wasm_tensor {
            assert_eq!(tensor_string.buffer, buffer);
        } else {
            return panic!("Expected WasmTensor::Str variant");
        }
    }

    #[test]
    fn wasm_tensor_to_carton_tensor_i32() {
        let buffer = slice_to_bytes(&[1i32, 2, 3]);
        let tensor = WasmTensor::Numeric(TensorNumeric {
            buffer: buffer.to_vec(),
            dtype: Dtype::I32,
            shape: vec![3],
        });
        let carton_tensor: CartonTensor = tensor.into();
        if let CartonTensor::I32(storage) = carton_tensor {
            assert_eq!(storage.view().as_slice().unwrap(), &[1i32, 2, 3]);
        } else {
            return panic!("Expected CartonTensor::I32 variant");
        }
    }

    #[test]
    fn wasm_tensor_to_carton_tensor_u32() {
        let buffer = slice_to_bytes(&[1u32, 2, 3]);
        let tensor = WasmTensor::Numeric(TensorNumeric {
            buffer: buffer.to_vec(),
            dtype: Dtype::Ui32,
            shape: vec![3],
        });
        let carton_tensor: CartonTensor = tensor.into();
        if let CartonTensor::U32(storage) = carton_tensor {
            assert_eq!(storage.view().as_slice().unwrap(), &[1u32, 2, 3]);
        } else {
            return panic!("Expected CartonTensor::U32 variant");
        }
    }
}