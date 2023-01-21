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
