use std::path::PathBuf;

use bytes::Buf;
use flate2::read::GzDecoder;
use tar::Archive;

struct PythonVersion {
    major: u32,
    minor: u32,
    micro: u32,
    url: &'static str,
}

#[tokio::main]
async fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let bundled_python_dir = manifest_dir.join("bundled_python");
    let python_configs_dir = manifest_dir.join("python_configs");
    std::fs::create_dir_all(&python_configs_dir).unwrap();

    let to_fetch = [PythonVersion {
        major: 3,
        minor: 10,
        micro: 9,
        url: "https://github.com/indygreg/python-build-standalone/releases/download/20230116/cpython-3.10.9+20230116-x86_64-unknown-linux-gnu-install_only.tar.gz",
    }];

    for PythonVersion {
        major,
        minor,
        micro,
        url,
    } in to_fetch
    {
        let out_dir = bundled_python_dir.join(format!("python{major}.{minor}.{micro}"));

        if !out_dir.exists() {
            std::fs::create_dir_all(&out_dir).unwrap();

            // Get the standalone python interpreter
            let res = reqwest::get(url).await.unwrap().bytes().await.unwrap();

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
