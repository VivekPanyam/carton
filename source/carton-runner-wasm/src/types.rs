use serde::{Deserialize, Serialize};
use carton_runner_interface::types::{Tensor, TensorStorage};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum DType {
    F32,
    F64,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
}

impl DType {
    pub fn mem_size(&self) -> usize {
        match self {
            DType::F32 => std::mem::size_of::<f32>(),
            DType::F64 => std::mem::size_of::<f64>(),
            DType::I8 => std::mem::size_of::<i8>(),
            DType::I16 => std::mem::size_of::<i16>(),
            DType::I32 => std::mem::size_of::<i32>(),
            DType::I64 => std::mem::size_of::<i64>(),
            DType::U8 => std::mem::size_of::<u8>(),
            DType::U16 => std::mem::size_of::<u16>(),
            DType::U32 => std::mem::size_of::<u32>(),
            DType::U64 => std::mem::size_of::<u64>(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OutputMetadata {
    pub dtype: DType,
    pub shape: Vec<u64>,
}

impl OutputMetadata {
    pub fn mem_size(&self) -> usize {
        self.shape.iter().product::<usize>() * self.dtype.mem_size()
    }

    pub fn make_tensor(&self) -> (Tensor, &'_ mut [u8]) {
        match self.dtype {
            DType::F32 => new_tensor_and_slice::<f32>(),
            DType::F64 => new_tensor_and_slice::<f64>(),
            DType::I8 => new_tensor_and_slice::<i8>(),
            DType::I16 => new_tensor_and_slice::<i16>(),
            DType::I32 => new_tensor_and_slice::<i32>(),
            DType::I64 => new_tensor_and_slice::<i64>(),
            DType::U8 => new_tensor_and_slice::<u8>(),
            DType::U16 => new_tensor_and_slice::<u16>(),
            DType::U32 => new_tensor_and_slice::<u32>(),
            DType::U64 => new_tensor_and_slice::<u64>(),
        }
    }
}

pub(crate) fn to_byte_slice<T>(s: &[T]) -> &'_ [u8] {
    let ptr = s.as_ptr() as *const u8;
    let len = s.len() * std::mem::size_of::<T>();
    unsafe { std::slice::from_raw_parts(ptr, len) }
}

pub(crate) fn new_tensor_and_slice<T>() -> (Tensor, &'_ mut [u8]) {
    let t = TensorStorage::<T>::new(vec![1]);
    (t.into(), &mut to_byte_slice(t.view().to_slice_memory_order().unwrap()))
}