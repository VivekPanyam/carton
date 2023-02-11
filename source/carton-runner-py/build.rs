use std::path::PathBuf;

fn main() {
    // Find the python lib dir to add to LD_LIBRARY_PATH
    let config_file_path =
        PathBuf::from(std::env::var("PYO3_CONFIG_FILE").expect("PYO3_CONFIG_FILE should be set"));
    let config = std::fs::read_to_string(config_file_path).unwrap();
    let libdir = config
        .lines()
        .into_iter()
        .find_map(|line| line.strip_prefix("lib_dir="))
        .unwrap();

    println!("cargo:rerun-if-env-changed=PYO3_CONFIG_FILE");
    println!("cargo:rustc-env=LD_LIBRARY_PATH={libdir}");
}
