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

use std::path::PathBuf;

fn main() {
    // Rerun only if this file changes (otherwise cargo would rerun this build script all the time)
    println!("cargo:rerun-if-changed=build.rs");

    // This is a bit hacky, but we need to set LD_LIBRARY_PATH at runtime for cargo test
    // We can't set this in .cargo/config.toml so we do it here
    // TODO: see if we can improve this
    let libdir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("xla_extension")
        .join("lib");

    #[cfg(not(target_os = "macos"))]
    println!(
        "cargo:rustc-env=LD_LIBRARY_PATH={}",
        libdir.to_str().unwrap()
    );

    #[cfg(target_os = "macos")]
    println!(
        "cargo:rustc-env=DYLD_FALLBACK_LIBRARY_PATH={}",
        libdir.to_str().unwrap()
    );

    // Add the bundled xla lib dir to the binary's rpath
    #[cfg(not(target_os = "macos"))]
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/xla_extension/lib");

    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path/xla_extension/lib");
}
