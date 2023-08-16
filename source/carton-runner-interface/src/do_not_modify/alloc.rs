use carton_macros::for_each_numeric_carton_type;

/// Numeric tensor types supported by this version of the runner interface
pub(crate) trait NumericTensorType: Default + Copy {}

for_each_numeric_carton_type! {
    $(
        impl NumericTensorType for $RustType {}
    )*
}

pub trait AsPtr<T> {
    /// Get a view of this tensor
    fn as_ptr(&self) -> *const T;

    /// Get a mut view of this tensor
    fn as_mut_ptr(&mut self) -> *mut T;
}

pub trait TypedAlloc<T> {
    type Output: AsPtr<T>;

    fn alloc(&self, numel: usize) -> Self::Output;
}
