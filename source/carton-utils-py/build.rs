fn main() {
    // Rerun only if this file changes (otherwise cargo would rerun this build script all the time)
    println!("cargo:rerun-if-changed=build.rs");

    #[cfg(target_os = "macos")]
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath,/Library/Developer/CommandLineTools/Library/Frameworks"
    );
}
