fn main() {
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/bundled_python/python/lib");
}
