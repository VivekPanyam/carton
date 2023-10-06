use color_eyre::{Report, Result};
use carton_runner_interface::types::{
    Tensor as CartonTensor,
};

use crate::component::{Dtype, Tensor};

impl TryFrom<CartonTensor> for Tensor {
	type Error = Report;

	fn try_from(value: CartonTensor) -> Result<Self> {
		todo!()
	}
}

impl Into<CartonTensor> for Tensor {
	fn into(self) -> CartonTensor {
		todo!()
	}
}

trait DTypeOf {
	fn dtype() -> Dtype;
}

macro_rules! for_each_numeric {
    ($macro_name:ident) => {
        $macro_name!(f32, Dtype::F32);
        $macro_name!(f64, Dtype::F64);
        $macro_name!(i32, Dtype::I32);
        $macro_name!(i64, Dtype::I64);
        $macro_name!(u8, Dtype::Ui8);
        $macro_name!(u16, Dtype::Ui16);
        $macro_name!(u32, Dtype::Ui32);
        $macro_name!(u64, Dtype::Ui64);
    };
}

macro_rules! implement_dtypeof {
    ($type:ty, $enum_variant:expr) => {
        impl DTypeOf for $type {
            fn dtype() -> Dtype {
                $enum_variant
            }
        }
    };
}

for_each_numeric!(implement_dtypeof);