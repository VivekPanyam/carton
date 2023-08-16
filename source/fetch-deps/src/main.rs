use std::path::PathBuf;

#[tokio::main]
pub async fn main() {
    fetch_libtorch().await;
}

/// Fetch libtorch
async fn fetch_libtorch() {
    let libtorch_dir = PathBuf::from(env!("LIBTORCH"));

    let url = fetch_deps::libtorch::URL;
    let sha256 = fetch_deps::libtorch::SHA256;

    if !libtorch_dir.exists() {
        println!("Downloading libtorch to {libtorch_dir:#?} from {url} ...");
        std::fs::create_dir_all(&libtorch_dir).unwrap();

        let td = tempfile::tempdir().unwrap();
        let download_path = td.path().join("download");

        // Download the file
        carton_utils::download::cached_download(url, sha256, &download_path, |_| {}, |_| {})
            .await
            .unwrap();

        // Unpack it (the zip file contains a libtorch dir so we unpack in the parent dir)
        carton_utils::archive::extract_zip(download_path.as_path(), libtorch_dir.parent().unwrap())
            .await;
    }
}
