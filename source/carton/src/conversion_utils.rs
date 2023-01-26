//! Utilities to convert between (Vec<T> -> Vec<U>) and (HashMap<_, T> -> HashMap<_, U>)
//! because `From` is not implemented for these types

use std::{collections::HashMap, hash::Hash};

pub(crate) fn convert_vec<T, U>(v: Vec<T>) -> Vec<U>
where
    U: From<T>,
{
    v.into_iter().map(|v| v.into()).collect()
}

pub(crate) fn convert_map<A: Hash + Eq, T, U>(v: HashMap<A, T>) -> HashMap<A, U>
where
    U: From<T>,
{
    v.into_iter().map(|(k, v)| (k, v.into())).collect()
}

pub(crate) fn convert_opt_vec<T, U>(v: Option<Vec<T>>) -> Option<Vec<U>>
where
    U: From<T>,
{
    v.map(|item| convert_vec(item))
}

pub(crate) fn convert_opt_map<A: Hash + Eq, T, U>(v: Option<HashMap<A, T>>) -> Option<HashMap<A, U>>
where
    U: From<T>,
{
    v.map(|item| convert_map(item))
}

/// Several useful conversions are not allowed by rust because they happen to overlap with the core
/// `impl<T> From<T> for T`.
///
/// For example, if we wanted to convert any tensor to a GenericTensor, we couldn't implement `From`
/// because it overlaps with converting a `GenericTensor` to a `GenericTensor` (i.e. From<T> for T)
///
/// Therefore, we create a separate conversion trait that is almost identical to From, but it doesn't
/// have an impl for From<T> for T
pub(crate) trait ConvertFrom<T> {
    fn from(value: T) -> Self;
}

// Something like "into"
pub(crate) trait ConvertInto<T> {
    fn convert_into(self) -> T;
}

// Blanket impl
impl<T, U> ConvertInto<U> for T
where
    U: ConvertFrom<T>,
{
    fn convert_into(self) -> U {
        U::from(self)
    }
}
