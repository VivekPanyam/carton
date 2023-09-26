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

use bytes::Buf;
use flate2::read::GzDecoder;
use sha2::{Digest, Sha256};
use tar::Archive;

fn main() {
    // Make sure we have python installed
    install_python();

    // Rerun only if this file changes (otherwise cargo would rerun this build script all the time)
    println!("cargo:rerun-if-changed=build.rs");

    // Add the bundled python lib dir to the binary's rpath
    #[cfg(not(target_os = "macos"))]
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/bundled_python/python/lib");

    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path/bundled_python/python/lib");

    #[cfg(target_os = "macos")]
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath,/Library/Developer/CommandLineTools/Library/Frameworks"
    );
}

// Include the list of python releases we want to build against
include!("src/bin/build_releases/python_versions.rs");

/// Install standalone python builds
#[tokio::main]
pub async fn install_python() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let bundled_python_dir = manifest_dir.join("bundled_python");
    let python_configs_dir = manifest_dir.join("python_configs");
    std::fs::create_dir_all(&python_configs_dir).unwrap();

    for PythonVersion {
        major,
        minor,
        micro,
        url,
        sha256,
    } in PYTHON_VERSIONS
    {
        println!("Fetching python {major}.{minor}.{micro} from {url} ({sha256})...");
        let out_dir = bundled_python_dir.join(format!("python{major}.{minor}.{micro}"));

        if !out_dir.exists() {
            std::fs::create_dir_all(&out_dir).unwrap();

            // Get the standalone python interpreter
            let res = reqwest::get(url).await.unwrap().bytes().await.unwrap();

            // Check the sha256
            let mut hasher = Sha256::new();
            hasher.update(&res);
            let actual_sha256 = format!("{:x}", hasher.finalize());
            assert_eq!(sha256, actual_sha256);

            // Unpack it
            let tar = GzDecoder::new(res.reader());
            let mut archive = Archive::new(tar);
            archive.unpack(&out_dir).unwrap();
        }

        let out_dir = out_dir.join("python/lib").canonicalize().unwrap();
        let lib_dir = out_dir.to_string_lossy();

        let config = format!(
            "implementation=CPython
version={major}.{minor}
shared=true
abi3=false
lib_name=python{major}.{minor}
lib_dir={lib_dir}
executable=
pointer_width=64
build_flags=
suppress_build_script_link_lines=false"
        );

        std::fs::write(
            python_configs_dir.join(format!("cpython{major}.{minor}.{micro}")),
            config,
        )
        .unwrap();
    }
}
