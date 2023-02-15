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
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/bundled_python/python/lib");
}

struct PythonVersion {
    major: u32,
    minor: u32,
    micro: u32,
    url: &'static str,
    sha256: &'static str,
}

/// Install standalone python builds
#[tokio::main]
pub async fn install_python() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let bundled_python_dir = manifest_dir.join("bundled_python");
    let python_configs_dir = manifest_dir.join("python_configs");
    std::fs::create_dir_all(&python_configs_dir).unwrap();

    let to_fetch = [
        PythonVersion {
            major: 3,
            minor: 10,
            micro: 9,
            url: "https://github.com/indygreg/python-build-standalone/releases/download/20230116/cpython-3.10.9+20230116-x86_64-unknown-linux-gnu-install_only.tar.gz",
            sha256: "d196347aeb701a53fe2bb2b095abec38d27d0fa0443f8a1c2023a1bed6e18cdf",
        },
        PythonVersion {
            major: 3,
            minor: 11,
            micro: 1,
            url: "https://github.com/indygreg/python-build-standalone/releases/download/20230116/cpython-3.11.1+20230116-x86_64-unknown-linux-gnu-install_only.tar.gz",
            sha256: "02a551fefab3750effd0e156c25446547c238688a32fabde2995c941c03a6423",
        },
        PythonVersion {
            major: 3,
            minor: 8,
            micro: 16,
            url: "https://github.com/indygreg/python-build-standalone/releases/download/20230116/cpython-3.8.16+20230116-x86_64-unknown-linux-gnu-install_only.tar.gz",
            sha256: "c890de112f1ae31283a31fefd2061d5c97bdd4d1bdd795552c7abddef2697ea1",
        },
        PythonVersion {
            major: 3,
            minor: 9,
            micro: 16,
            url: "https://github.com/indygreg/python-build-standalone/releases/download/20230116/cpython-3.9.16+20230116-x86_64-unknown-linux-gnu-install_only.tar.gz",
            sha256: "7ba397787932393e65fc2fb9fcfabf54f2bb6751d5da2b45913cb25b2d493758",
        },
    ];

    for PythonVersion {
        major,
        minor,
        micro,
        url,
        sha256,
    } in to_fetch
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
