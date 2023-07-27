//! Implementation of an enum that wraps all the runner tensor types and provides storage

use crate::types::{TensorStorage, TypedStorage};
use lunchbox::types::{MaybeSend, MaybeSync};

#[derive(Debug)]
pub struct RunnerStorage;

impl TensorStorage for RunnerStorage {
    type TypedStorage<T> = TypedRunnerStorage<T>
    where
        T: MaybeSend + MaybeSync;

    type TypedStringStorage = TypedRunnerStorage<String>;
}

#[derive(Debug)]
pub enum TypedRunnerStorage<T> {
    V1(runner_interface_v1::types::TensorStorage<T>),
}

impl<T> TypedStorage<T> for TypedRunnerStorage<T> {
    fn view(&self) -> ndarray::ArrayViewD<T> {
        match self {
            TypedRunnerStorage::V1(s) => s.view(),
        }
    }

    fn view_mut(&mut self) -> ndarray::ArrayViewMutD<T> {
        match self {
            TypedRunnerStorage::V1(s) => s.view_mut(),
        }
    }
}
