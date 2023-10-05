use serde::{Deserialize, Serialize};
use carton_runner_interface::types::{Tensor, TensorStorage};

pub(crate) fn to_byte_slice<T>(s: &[T]) -> &'static mut [u8] {
    let ptr = s.as_ptr() as *mut u8;
    let len = s.len() * std::mem::size_of::<T>();
    unsafe { std::slice::from_raw_parts_mut(ptr, len) }
}

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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OutputMetadata {
    pub dtype: DType,
    pub shape: Vec<u64>,
}

macro_rules! new_tensor_and_slice {
    ($shape:expr, $dtype:ty) => {{
        let t = TensorStorage::<$dtype>::new($shape);
        let p = to_byte_slice(t.view().to_slice_memory_order().unwrap());
        (t.into(), p)
    }};
}

impl OutputMetadata {
    pub fn mem_size(&self) -> usize {
        self.shape.iter().product::<u64>() as usize * self.dtype.mem_size()
    }

    pub fn make_tensor(&self) -> (Tensor, &'_ mut [u8]) {
        match self.dtype {
            DType::F32 => new_tensor_and_slice!(self.shape.clone(), f32),
            DType::F64 => new_tensor_and_slice!(self.shape.clone(), f64),
            DType::I8 => new_tensor_and_slice!(self.shape.clone(), i8),
            DType::I16 => new_tensor_and_slice!(self.shape.clone(), i16),
            DType::I32 => new_tensor_and_slice!(self.shape.clone(), i32),
            DType::I64 => new_tensor_and_slice!(self.shape.clone(), i64),
            DType::U8 => new_tensor_and_slice!(self.shape.clone(), u8),
            DType::U16 => new_tensor_and_slice!(self.shape.clone(), u16),
            DType::U32 => new_tensor_and_slice!(self.shape.clone(), u32),
            DType::U64 => new_tensor_and_slice!(self.shape.clone(), u64),
        }
    }
}