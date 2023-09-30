#[ocaml_rust::bridge]
mod ffi {
    extern "Rust" {
        fn add_one(x: isize) -> isize;
    }
}

fn add_one(x: isize) -> isize {
    x + 1
}
