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

use std::{path::PathBuf, process::Command};

#[cfg(not(target_os = "macos"))]
const LIBRARY_SUFFIX: &'static str = ".so";

#[cfg(target_os = "macos")]
const LIBRARY_SUFFIX: &'static str = ".dylib";

/// This test compiles all of the c files in this directory and tests them
#[test]
fn test_c_examples() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Build the bindings
    log::info!("Building dylib for C bindings...");
    let cargo_messages = escargot::CargoBuild::new()
        .package("carton-bindings-c")
        .current_release()
        .current_target()
        .arg("--timings")
        .exec()
        .unwrap();

    // Get the artifact path
    let cdylib_path = cargo_messages
        .filter_map(|ref message| {
            let message = message.as_ref().unwrap();
            let decoded = message.decode().unwrap();
            extract_cdylib(&decoded, "cdylib")
        })
        .next()
        .unwrap();

    // For each c file in the tests dir
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .read_dir()
        .unwrap();
    for entry in dir {
        let entry = entry.unwrap();
        let file_name = entry.file_name().to_str().unwrap().to_owned();
        if file_name.ends_with(".c") {
            // Compile the file
            let path = entry.path();
            log::info!("About to build and test C example: {path:?}");

            // cc builds libs, but we want to build an executable
            // This is based on https://github.com/Hywan/inline-c-rs
            let compiler = cc::Build::new()
                .cargo_metadata(false)
                .target(escargot::CURRENT_TARGET)
                .opt_level(1)
                .host(escargot::CURRENT_TARGET)
                .try_get_compiler()
                .unwrap();

            let tempdir = tempfile::tempdir().unwrap();

            let mut command = Command::new(compiler.path());
            command.arg(path);
            command.args(compiler.args());
            command.arg("-o").arg(tempdir.path().join("test"));
            command.arg("-L").arg(cdylib_path.parent().unwrap());
            command.arg("-l").arg(
                cdylib_path
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .strip_prefix("lib")
                    .unwrap()
                    .strip_suffix(LIBRARY_SUFFIX)
                    .unwrap(),
            );

            let mut compiler_output = command.spawn().unwrap();
            assert!(compiler_output.wait().unwrap().success());

            // Run the compiled executable
            let mut command = Command::new(tempdir.path().join("test")).spawn().unwrap();
            assert!(
                command.wait().unwrap().success(),
                "Test {file_name} failed."
            );
        }
    }
}

/// Based on `extract_bin` within escargot
fn extract_cdylib(msg: &escargot::format::Message, desired_kind: &str) -> Option<PathBuf> {
    match msg {
        escargot::format::Message::CompilerArtifact(art) => {
            if !art.profile.test
                && art.target.crate_types == ["cdylib"]
                && art.target.kind == [desired_kind]
            {
                Some(
                    art.filenames
                        .get(0)
                        .expect("files must exist")
                        .to_path_buf(),
                )
            } else {
                None
            }
        }
        _ => None,
    }
}
