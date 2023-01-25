use std::{fmt::Debug, marker::PhantomData};

use ndarray::{ShapeBuilder, StrideShape};
use serde::{Deserialize, Serialize};

pub trait Storage<T>: Debug {
    fn get(&self) -> &[T];
}

impl<T, U> Storage<T> for U
where
    U: AsRef<[T]> + Debug,
{
    fn get(&self) -> &[T] {
        self.as_ref()
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NDarray<T, S> {
    shape: Vec<u64>,
    strides: Option<Vec<i64>>,

    storage: S,
    _pd: PhantomData<T>,
}

impl<T, S> NDarray<T, S>
where
    S: Storage<T>,
{
    pub fn from_shape_storage(shape: Vec<u64>, storage: S) -> Self {
        Self {
            shape,
            strides: None,
            storage,
            _pd: PhantomData,
        }
    }

    pub fn from_shape_strides_storage(
        shape: Vec<u64>,
        strides: Option<Vec<i64>>,
        storage: S,
    ) -> Self {
        Self {
            shape,
            strides,
            storage,
            _pd: PhantomData,
        }
    }

    pub fn shape(&self) -> &Vec<u64> {
        &self.shape
    }

    pub fn strides(&self) -> &Option<Vec<i64>> {
        &self.strides
    }

    pub fn into_storage(self) -> S {
        self.storage
    }

    /// Data laid out according to `strides` and `shape`
    pub fn data(&self) -> &[T] {
        self.storage.get()
    }

    pub fn data_mut(&self) -> &mut [T] {
        let data = self.data();

        // TODO: does this break anything?
        unsafe { std::slice::from_raw_parts_mut(data.as_ptr() as _, data.len()) }
    }

    fn get_shape(&self) -> StrideShape<ndarray::IxDyn> {
        match &self.strides {
            None => self
                .shape
                .iter()
                .map(|v| *v as usize)
                .collect::<Vec<_>>()
                .into(),
            Some(strides) => self
                .shape
                .iter()
                .map(|v| *v as usize)
                .collect::<Vec<_>>()
                .strides(strides.iter().map(|v| (*v).try_into().unwrap()).collect())
                .into(),
        }
    }

    pub fn view(&self) -> ndarray::ArrayViewD<T> {
        let data = self.storage.get();
        unsafe { ndarray::ArrayView::from_shape_ptr(self.get_shape(), data.as_ptr() as _) }
    }

    pub fn view_mut(&self) -> ndarray::ArrayViewMutD<T> {
        let data = self.storage.get();
        // TODO: does this *const T to *mut T cast break anything?
        unsafe { ndarray::ArrayViewMut::from_shape_ptr(self.get_shape(), data.as_ptr() as _) }
    }
}
