// Copyright 2023 Vivek Panyam
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::sync::OnceLock;

use tokio::runtime::Runtime;

/// A utility to lazily start a tokio runtime
pub(crate) fn runtime() -> &'static Runtime {
    static CELL: OnceLock<Runtime> = OnceLock::new();
    CELL.get_or_init(|| Runtime::new().unwrap())
}

/// A macro that helps ensure that we (somewhat safely) convert between C and Rust types
macro_rules! ffi_conversions {
    ($t:ident) => {
        impl From<Box<$t>> for *mut $t {
            fn from(value: Box<$t>) -> Self {
                // SAFETY: We use Box::from_raw below
                Box::into_raw(value)
            }
        }

        impl From<*mut $t> for Box<$t> {
            fn from(value: *mut $t) -> Self {
                // SAFETY: We ues Box::into_raw above
                unsafe { Box::from_raw(value) }
            }
        }
    };
}
