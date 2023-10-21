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

use std::{
    path::{Path, PathBuf},
    process::Command,
};

use build_utils::{build_c_bindings, build_cpp_bindings};
use clap::Parser;

#[derive(Parser, Debug)]
struct Args {
    /// The local folder to output bindings too
    #[arg(long)]
    bindings_path: PathBuf,

    /// The compilation target (used as a suffix for the bindings tar)
    #[arg(long)]
    target: String,
}

fn main() {
    // Logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Parse args
    let args = Args::parse();

    build_c_cpp_bindings(&args.bindings_path, &args.target);
}

fn build_c_cpp_bindings(bindings_path: &Path, target: &str) {
    let source_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf();

    let tempdir = tempfile::tempdir().unwrap();

    // Build the C bindings and copy them into the output dir
    let c_dir = tempdir.path().join("c");
    std::fs::create_dir(&c_dir).unwrap();

    // Create lib and include dirs
    let c_lib_dir = c_dir.join("lib");
    let c_include_dir = c_dir.join("include");
    std::fs::create_dir(&c_lib_dir).unwrap();
    std::fs::create_dir(&c_include_dir).unwrap();

    // Build the C bindings and copy them to the lib dir
    let c_bindings = build_c_bindings();
    std::fs::copy(c_bindings.shared_lib, c_lib_dir.join("libcarton_c.so")).unwrap();
    std::fs::copy(c_bindings.static_lib, c_lib_dir.join("libcarton_c.a")).unwrap();

    // Copy the header to the include dir
    std::fs::copy(
        source_dir.join("carton-bindings-c/carton.h"),
        c_include_dir.join("carton.h"),
    )
    .unwrap();

    // Make a tar file for the C bindings
    assert!(Command::new("tar")
        .arg("-cvzf")
        .arg(
            bindings_path
                .canonicalize()
                .unwrap()
                .join(format!("carton_c_{}.tar.gz", target))
        )
        .arg(".")
        .current_dir(c_dir)
        .status()
        .unwrap()
        .success());

    // Build the C++ bindings and copy them into the output dir
    let cpp_dir = tempdir.path().join("cpp");
    std::fs::create_dir(&cpp_dir).unwrap();

    // Create lib and include dirs
    let cpp_lib_dir = cpp_dir.join("lib");
    let cpp_include_dir = cpp_dir.join("include");
    std::fs::create_dir(&cpp_lib_dir).unwrap();
    std::fs::create_dir(&cpp_include_dir).unwrap();

    // Build the C++ bindings and put the files in the lib dir
    build_cpp_bindings(&cpp_lib_dir);

    // Copy header files in
    std::fs::copy(
        source_dir.join("carton-bindings-cpp/src/carton.hh"),
        &cpp_include_dir.join("carton.hh"),
    )
    .unwrap();
    std::fs::copy(
        source_dir.join("carton-bindings-cpp/src/carton_impl.hh"),
        cpp_include_dir.join("carton_impl.hh"),
    )
    .unwrap();

    // Make a tar file for the C++ bindings
    assert!(Command::new("tar")
        .arg("-cvzf")
        .arg(
            bindings_path
                .canonicalize()
                .unwrap()
                .join(format!("carton_cpp_{}.tar.gz", target))
        )
        .arg(".")
        .current_dir(cpp_dir)
        .status()
        .unwrap()
        .success());
}
