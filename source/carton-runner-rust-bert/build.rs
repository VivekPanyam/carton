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
        .join("libtorch")
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

    // Add the bundled libtorch lib dir to the binary's rpath
    #[cfg(not(target_os = "macos"))]
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/libtorch/lib");

    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path/libtorch/lib");
}
