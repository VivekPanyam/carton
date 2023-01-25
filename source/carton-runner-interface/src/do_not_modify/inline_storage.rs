use crate::types::NDarray;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

use crate::do_not_modify::ndarray::Storage;

#[derive(Debug, Serialize, Deserialize)]
pub struct TensorStorage<T> {
    data: Vec<T>,
}

impl<T> Storage<T> for TensorStorage<T>
where
    T: Debug,
{
    fn get(&self) -> &[T] {
        &self.data
    }
}

// Copy the data
impl<T, S> From<NDarray<T, S>> for NDarray<T, TensorStorage<T>>
where
    T: Debug + Clone + Copy + Default,
    S: AsRef<[T]> + Debug,
{
    fn from(value: NDarray<T, S>) -> Self {
        let view = value.view();

        let out = alloc_tensor(view.shape().iter().map(|v| (*v) as _).collect());

        if view.is_standard_layout() {
            // We can just memcpy the data
            out.data_mut().copy_from_slice(value.data())
        } else {
            out.view_mut().assign(&value.view());
        }

        out
    }
}

// Allocates a contiguous tensor with a shape and type
pub(crate) fn alloc_tensor<T: Debug + std::clone::Clone + std::default::Default>(
    shape: Vec<u64>,
) -> NDarray<T, TensorStorage<T>> {
    let numel: u64 = shape.iter().product();
    let data: Vec<T> = vec![T::default(); numel as _];
    NDarray::from_shape_storage(shape, TensorStorage { data })
}
