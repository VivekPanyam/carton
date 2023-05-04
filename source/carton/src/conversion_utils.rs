//! Utilities to convert between (Vec<T> -> Vec<U>) and (HashMap<_, T> -> HashMap<_, U>)
//! because `From` is not implemented for these types

use std::{collections::HashMap, hash::Hash};

pub fn convert_vec<T, U>(v: Vec<T>) -> Vec<U>
where
    U: From<T>,
{
    v.into_iter().map(|v| v.into()).collect()
}

pub fn convert_map<A: Hash + Eq, T, U>(v: HashMap<A, T>) -> HashMap<A, U>
where
    U: From<T>,
{
    v.into_iter().map(|(k, v)| (k, v.into())).collect()
}

pub fn convert_opt_vec<T, U>(v: Option<Vec<T>>) -> Option<Vec<U>>
where
    U: From<T>,
{
    v.map(|item| convert_vec(item))
}

pub fn convert_opt_map<A: Hash + Eq, T, U>(v: Option<HashMap<A, T>>) -> Option<HashMap<A, U>>
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
///
/// Note: the trait that should be implemented is `ConvertFromWithContext`
pub(crate) trait ConvertFrom<T>: ConvertFromWithContext<T, ()> + Sized {
    fn from(value: T) -> Self {
        <Self as ConvertFromWithContext<T, ()>>::from(value, ())
    }
}

/// Blanket impl
impl<T, U> ConvertFrom<T> for U where U: ConvertFromWithContext<T, ()> {}

/// Allows conversions with context
///
/// Implementors should propagate `context` into any conversions they use internally.
/// `ConvertFrom` will automatically be implemented for this type if the context is unnecessary
/// (i.e. no transitive conversions require context)
pub(crate) trait ConvertFromWithContext<T, C>
where
    C: Copy,
{
    fn from(value: T, context: C) -> Self;
}

/// Something like "into"
pub(crate) trait ConvertInto<T>
where
    T: ConvertFrom<Self>,
    Self: Sized,
{
    fn convert_into(self) -> T {
        <T as ConvertFrom<Self>>::from(self)
    }
}

/// Blanket impl
impl<T, U> ConvertInto<T> for U where T: ConvertFrom<U> {}

/// Something like "into", but with context
pub(crate) trait ConvertIntoWithContext<T, C>
where
    T: ConvertFromWithContext<Self, C>,
    Self: Sized,
    C: Copy,
{
    fn convert_into_with_context(self, context: C) -> T {
        T::from(self, context)
    }
}

/// Blanket impl
impl<T, U, C> ConvertIntoWithContext<T, C> for U
where
    T: ConvertFromWithContext<U, C>,
    C: Copy,
{
}
