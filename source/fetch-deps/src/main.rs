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

#[tokio::main]
pub async fn main() {
    let _ = tokio::join!(fetch_libtorch(), fetch_xla());
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
        carton_utils::download::cached_download(
            url,
            sha256,
            Some(&download_path),
            None,
            |_| {},
            |_| {},
        )
        .await
        .unwrap();

        // Unpack it (the zip file contains a libtorch dir so we unpack in the parent dir)
        carton_utils::archive::extract_zip(download_path.as_path(), libtorch_dir.parent().unwrap())
            .await;
    }
}

/// Fetch xla
async fn fetch_xla() {
    let xla_dir = PathBuf::from(env!("XLA_EXTENSION_DIR"));

    let url = fetch_deps::xla::URL;
    let sha256 = fetch_deps::xla::SHA256;

    if !xla_dir.exists() {
        println!("Downloading XLA to {xla_dir:#?} from {url} ...");
        std::fs::create_dir_all(&xla_dir).unwrap();

        let td = tempfile::tempdir().unwrap();
        let download_path = td.path().join("download");

        // Download the file
        carton_utils::download::cached_download(
            url,
            sha256,
            Some(&download_path),
            None,
            |_| {},
            |_| {},
        )
        .await
        .unwrap();

        // Unpack it (the tar file contains a xla_extension dir so we unpack in the parent dir)
        carton_utils::archive::extract_tar_gz(download_path.as_path(), xla_dir.parent().unwrap())
            .await;
    }
}
