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

pub struct CBindings {
    pub shared_lib: PathBuf,
    pub static_lib: PathBuf,
}

/// Build the Carton C bindings
pub fn build_c_bindings() -> CBindings {
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

            match decoded {
                escargot::format::Message::CompilerArtifact(art) => {
                    if !art.profile.test
                        && art.target.name == "carton-bindings-c"
                        && art.target.crate_types == ["staticlib", "cdylib"]
                        && art.target.kind == ["staticlib", "cdylib"]
                    {
                        if art
                            .filenames
                            .get(0)
                            .unwrap()
                            .extension()
                            .unwrap()
                            .to_str()
                            .unwrap()
                            == "so"
                        {
                            // Shared lib first
                            Some(CBindings {
                                shared_lib: art.filenames.get(0).unwrap().to_path_buf(),
                                static_lib: art.filenames.get(1).unwrap().to_path_buf(),
                            })
                        } else {
                            // Static lib first
                            Some(CBindings {
                                shared_lib: art.filenames.get(1).unwrap().to_path_buf(),
                                static_lib: art.filenames.get(0).unwrap().to_path_buf(),
                            })
                        }
                    } else {
                        None
                    }
                }
                _ => None,
            }
        })
        .unwrap()
}

/// Build the Carton C++ bindings
pub fn build_cpp_bindings(output_folder: &Path) {
    let c_bindings_path = build_c_bindings().static_lib;
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

    // Build a .o file
    let tempdir = tempfile::tempdir().unwrap();
    let mut command = Command::new(compiler.path());
    command
        .arg(manifest_dir.join("../carton-bindings-cpp/src/carton.cc"))
        .args(compiler.args())
        .arg("-std=c++20")
        .arg("-I")
        .arg(manifest_dir.join("../carton-bindings-c"))
        .arg("-c")
        .arg("-o")
        .arg(tempdir.path().join("cartoncpp.o"));

    log::info!("Running command {command:?}");

    let mut compiler_output = command.spawn().unwrap();
    assert!(compiler_output.wait().unwrap().success());

    // Build a static library
    // TODO: this isn't ideal because it requires ar on the path
    std::fs::copy(&c_bindings_path, output_folder.join("libcarton_cpp.a")).unwrap();
    let mut command = Command::new("ar");
    command
        .arg("-rv")
        .arg(output_folder.join("libcarton_cpp.a"))
        .arg(tempdir.path().join("cartoncpp.o"));

    log::info!("Running command {command:?}");

    let mut ar_output = command.spawn().unwrap();
    assert!(ar_output.wait().unwrap().success());

    // Build a shared library
    let mut command = Command::new(compiler.path());
    command
        .arg("-shared")
        .arg("-o")
        .arg(output_folder.join("libcarton_cpp.so"))
        .arg(tempdir.path().join("cartoncpp.o"))
        .arg(c_bindings_path);

    #[cfg(not(target_os = "macos"))]
    command.arg("-pthread").arg("-ldl");

    #[cfg(target_os = "macos")]
    command
        .arg("-framework")
        .arg("CoreFoundation")
        .arg("-framework")
        .arg("Security");

    log::info!("Running command {command:?}");

    let mut compiler_output = command.spawn().unwrap();
    assert!(compiler_output.wait().unwrap().success());
}
