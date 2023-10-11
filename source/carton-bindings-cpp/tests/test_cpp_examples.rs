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

use std::{path::PathBuf, process::Command, time::Instant};

#[test]
fn test_cpp_examples() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Build the bindings
    let bindings_dir = tempfile::tempdir().unwrap();
    let bindings_path = bindings_dir.path().join("libcarton_cpp.so");
    build_utils::build_cpp_bindings(&bindings_path);

    // For each cc file in the tests dir
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .read_dir()
        .unwrap();
    for entry in dir {
        let entry = entry.unwrap();
        let file_name = entry.file_name().to_str().unwrap().to_owned();
        if file_name.ends_with(".cc") {
            // Compile the file
            let path = entry.path();
            log::info!("About to build and test C++ example: {path:?}");

            // cc builds libs, but we want to build an executable
            // This is based on https://github.com/Hywan/inline-c-rs
            let compiler = cc::Build::new()
                .cpp(true)
                .cargo_metadata(false)
                .target(escargot::CURRENT_TARGET)
                .opt_level(1)
                .host(escargot::CURRENT_TARGET)
                .try_get_compiler()
                .unwrap();

            let tempdir = tempfile::tempdir().unwrap();

            let mut command = Command::new(compiler.path());
            command.arg("-std=c++20");
            command.arg(path);
            command.arg(bindings_path.as_path());
            command.args(compiler.args());
            command.arg("-o").arg(tempdir.path().join("test"));
            command.arg("-lm");

            let mut compiler_output = command.spawn().unwrap();
            assert!(compiler_output.wait().unwrap().success());

            // Run the compiled executable
            let start = Instant::now();
            let mut command = Command::new(tempdir.path().join("test")).spawn().unwrap();
            assert!(
                command.wait().unwrap().success(),
                "Test {file_name} failed."
            );
            log::info!("Test {file_name} passed in {:?}", start.elapsed());
        }
    }
}
