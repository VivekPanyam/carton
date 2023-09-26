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

use proc_macro::TokenStream;
use quote::quote;

// Nested repeating macros get complex with declarative macros
// so we'll use a proc macro instead
// https://github.com/rust-lang/rust/issues/35853
#[proc_macro]
pub fn for_each_carton_type(item: TokenStream) -> TokenStream {
    let item = proc_macro2::TokenStream::from(item);
    quote! {

            // Declare the inner macro
            macro_rules! inner {
                ($( ( $CartonType:ident, $RustType:ty, $TypeStr:literal ) ), * ) => {
                    #item
                };
            }

            // Call it for each type
            inner!(
                (Float, f32, "float32"),
                (Double, f64, "float64"),
                (String, String, "string"),
                (I8, i8, "int8"),
                (I16, i16, "int16"),
                (I32, i32, "int32"),
                (I64, i64, "int64"),
                (U8, u8, "uint8"),
                (U16, u16, "uint16"),
                (U32, u32, "uint32"),
                (U64, u64, "uint64")
            );
    }
    .into()
}

#[proc_macro]
pub fn for_each_numeric_carton_type(item: TokenStream) -> TokenStream {
    let item = proc_macro2::TokenStream::from(item);
    quote! {

            // Declare the inner macro
            macro_rules! inner {
                ($( ( $CartonType:ident, $RustType:ty, $TypeStr:literal ) ), * ) => {
                    #item
                };
            }

            // Call it for each type
            inner!(
                (Float, f32, "float32"),
                (Double, f64, "float64"),
                (I8, i8, "int8"),
                (I16, i16, "int16"),
                (I32, i32, "int32"),
                (I64, i64, "int64"),
                (U8, u8, "uint8"),
                (U16, u16, "uint16"),
                (U32, u32, "uint32"),
                (U64, u64, "uint64")
            );
    }
    .into()
}
