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

/// Build the Carton C bindings
pub fn build_c_bindings() -> PathBuf {
    // Build the bindings
    log::info!("Building C bindings...");
    let mut cargo_messages = escargot::CargoBuild::new()
        .package("carton-bindings-c")
        .current_release()
        .current_target()
        .arg("--timings")
        .exec()
        .unwrap();

    // Get the artifact path
    cargo_messages
        .find_map(|ref message| {
            let message = message.as_ref().unwrap();
            let decoded = message.decode().unwrap();
            extract_lib(&decoded, "staticlib")
        })
        .unwrap()
}

/// Build the Carton C++ bindings
pub fn build_cpp_bindings(output_path: &Path) {
    let c_bindings_path = build_c_bindings();
    log::info!("Building C++ bindings...");
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    // This is based on https://github.com/Hywan/inline-c-rs
    let compiler = cc::Build::new()
        .cpp(true)
        .cargo_metadata(false)
        .target(escargot::CURRENT_TARGET)
        .opt_level(3)
        .host(escargot::CURRENT_TARGET)
        .try_get_compiler()
        .unwrap();

    let mut command = Command::new(compiler.path());
    command.arg(manifest_dir.join("../carton-bindings-cpp/src/carton.cc"));
    command.arg(c_bindings_path);
    command.args(compiler.args());
    command.arg("-std=c++20");
    command
        .arg("-I")
        .arg(manifest_dir.join("../carton-bindings-c"));
    command.arg("-shared");

    #[cfg(not(target_os = "macos"))]
    command.arg("-pthread").arg("-ldl");

    #[cfg(target_os = "macos")]
    command
        .arg("-framework")
        .arg("CoreFoundation")
        .arg("-framework")
        .arg("Security");

    command.arg("-o").arg(output_path);

    log::info!("Running command {command:?}");

    let mut compiler_output = command.spawn().unwrap();
    assert!(compiler_output.wait().unwrap().success());
}

/// Based on `extract_bin` within escargot
fn extract_lib(msg: &escargot::format::Message, desired_kind: &str) -> Option<PathBuf> {
    match msg {
        escargot::format::Message::CompilerArtifact(art) => {
            if !art.profile.test
                && art.target.crate_types == [desired_kind]
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
