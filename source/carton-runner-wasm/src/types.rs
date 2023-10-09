use color_eyre::eyre::{ensure, eyre};
use color_eyre::{Report, Result};

use carton::types::for_each_numeric_carton_type;
use carton_runner_interface::types::{Tensor as CartonTensor, TensorStorage as CartonStorage};

use crate::component::{Dtype, Tensor as WasmTensor, TensorNumeric, TensorString};

impl Into<CartonTensor> for WasmTensor {
    fn into(self) -> CartonTensor {
        match self {
            WasmTensor::Numeric(t) => t.into(),
            WasmTensor::String(t) => t.into(),
        }
    }
}

fn bytes_to_slice<T>(b: &[u8]) -> Result<&[T]> {
    ensure!(
        b.len() % std::mem::size_of::<T>() == 0,
        "Invalid byte length"
    );
    Ok(unsafe {
        std::slice::from_raw_parts(b.as_ptr() as *const T, b.len() / std::mem::size_of::<T>())
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
            Dtype::Float => {
                copy_to_storage(CartonStorage::<f32>::new(self.shape), &self.buffer).into()
            }
            Dtype::Double => {
                copy_to_storage(CartonStorage::<f64>::new(self.shape), &self.buffer).into()
            }
            Dtype::I8 => copy_to_storage(CartonStorage::<i8>::new(self.shape), &self.buffer).into(),
            Dtype::I16 => {
                copy_to_storage(CartonStorage::<i16>::new(self.shape), &self.buffer).into()
            }
            Dtype::I32 => {
                copy_to_storage(CartonStorage::<i32>::new(self.shape), &self.buffer).into()
            }
            Dtype::I64 => {
                copy_to_storage(CartonStorage::<i64>::new(self.shape), &self.buffer).into()
            }
            Dtype::U8 => copy_to_storage(CartonStorage::<u8>::new(self.shape), &self.buffer).into(),
            Dtype::U16 => {
                copy_to_storage(CartonStorage::<u16>::new(self.shape), &self.buffer).into()
            }
            Dtype::U32 => {
                copy_to_storage(CartonStorage::<u32>::new(self.shape), &self.buffer).into()
            }
            Dtype::U64 => {
                copy_to_storage(CartonStorage::<u64>::new(self.shape), &self.buffer).into()
            }
        }
    }
}

impl Into<CartonTensor> for TensorString {
    fn into(self) -> CartonTensor {
        let mut t = CartonStorage::new(self.shape);
        t.view_mut()
            .as_slice_mut()
            .unwrap()
            .clone_from_slice(&self.buffer);
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
            CartonTensor::String(t) => WasmTensor::String(t.into()),
            CartonTensor::NestedTensor(_) => return Err(eyre!("Nested tensors are not supported")),
        })
    }
}

fn slice_to_bytes<T>(s: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(s.as_ptr() as *const u8, s.len() * std::mem::size_of::<T>())
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
        TensorString { buffer, shape }
    }
}

trait DTypeOf {
    fn dtype() -> Dtype;
}

for_each_numeric_carton_type! {
    $(
        impl DTypeOf for $RustType {
            fn dtype() -> Dtype {
                Dtype::$CartonType
            }
        }
    )*
}

#[cfg(test)]
mod tests {
    use super::*;

    for_each_numeric_carton_type! {
        $(
            paste::item! {
                #[test]
                fn [< $TypeStr "_tensor_carton_to_wasm" >]() {
                    let storage = CartonStorage::<$RustType>::new(vec![3]);
                    let carton_tensor = CartonTensor::$CartonType(
                        copy_to_storage(
                            storage,
                            slice_to_bytes(
                                &[1.0 as $RustType, 2.0 as $RustType, 3.0 as $RustType]
                            )
                        )
                    );
                    let wasm_tensor = WasmTensor::try_from(carton_tensor).unwrap();
                    match wasm_tensor {
                        WasmTensor::Numeric(tensor_numeric) => {
                            assert_eq!(
                                tensor_numeric.buffer,
                                slice_to_bytes(&[1.0 as $RustType, 2.0 as $RustType, 3.0 as $RustType])
                            );
                        }
                        _ => {
                            panic!(concat!("Expected WasmTensor::Numeric variant"));
                        }
                    }
                }

                #[test]
                fn [< $TypeStr "_tensor_wasm_to_carton" >]() {
                    let buffer = slice_to_bytes(&[1.0 as $RustType, 2.0 as $RustType, 3.0 as $RustType]);
                    let tensor = WasmTensor::Numeric(TensorNumeric {
                        buffer: buffer.to_vec(),
                        dtype: Dtype::$CartonType,
                        shape: vec![3],
                    });
                    let carton_tensor: CartonTensor = tensor.into();
                    match carton_tensor {
                        CartonTensor::$CartonType(storage) => {
                            assert_eq!(
                                storage.view().as_slice().unwrap(),
                                &[1.0 as $RustType, 2.0 as $RustType, 3.0 as $RustType]
                            );
                        }
                        _ => {
                            panic!(concat!("Expected CartonTensor::", stringify!($CartonType), " variant"));
                        }
                    }
                }
            }
        )*
    }

    #[test]
    fn string_tensor_carton_to_wasm() {
        let buffer = vec!["hello".to_string(), "world".to_string()];
        let mut storage = CartonStorage::<String>::new(vec![2]);
        storage
            .view_mut()
            .as_slice_mut()
            .unwrap()
            .clone_from_slice(&buffer);
        let carton_tensor = CartonTensor::String(storage);
        let wasm_tensor = WasmTensor::try_from(carton_tensor).unwrap();
        match wasm_tensor {
            WasmTensor::String(tensor_string) => {
                assert_eq!(tensor_string.buffer, buffer);
            }
            _ => {
                panic!("Expected WasmTensor::String variant");
            }
        }
    }

    #[test]
    fn string_tensor_wasm_to_carton() {
        let buffer = vec!["hello".to_string(), "world".to_string()];
        let tensor = WasmTensor::String(TensorString {
            buffer: buffer.clone(),
            shape: vec![2],
        });
        let carton_tensor: CartonTensor = tensor.into();
        match carton_tensor {
            CartonTensor::String(storage) => {
                assert_eq!(storage.view().as_slice().unwrap(), &buffer);
            }
            _ => {
                panic!("Expected CartonTensor::String variant");
            }
        }
    }
}
