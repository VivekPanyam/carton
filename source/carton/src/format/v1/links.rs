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

use std::collections::HashMap;

use lunchbox::{
    path::{LunchboxPathUtils, PathBuf},
    ReadableFileSystem,
};
use serde::{Deserialize, Serialize};
use zipfs::ZipFS;

use crate::error::CartonError;

#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub(crate) struct Links {
    /// The version of the links file. Should be 1
    pub(crate) version: u64,

    /// Map from a sha256 to a vec of URLs
    pub(crate) urls: HashMap<String, Vec<String>>,
}

impl From<Vec<crate::info::LinkedFile>> for Links {
    fn from(value: Vec<crate::info::LinkedFile>) -> Self {
        let mut urls = HashMap::new();
        for item in value {
            urls.insert(item.sha256, item.urls);
        }
        Links { version: 1, urls }
    }
}

/// Take a path to a packed carton along with a map from sha256 to urls and shrink the carton by storing
/// URLs instead of the orig files when possible
#[cfg(not(target_family = "wasm"))]
pub(crate) async fn create_links(
    path: std::path::PathBuf,
    urls: HashMap<String, Vec<String>>,
) -> crate::error::Result<std::path::PathBuf> {
    use std::io::Write;

    let fs = ZipFS::new(path).await;

    let has_links = PathBuf::from("/LINKS").exists(&fs).await;

    // Load the file if there is one
    let mut links = if has_links {
        let links = fs.read_to_string("/LINKS").await?;
        toml::from_str(&links)?
    } else {
        Links {
            version: 1,
            urls: HashMap::new(),
        }
    };

    // Add URLs to links
    for (sha256, mut urls) in urls {
        links.urls.entry(sha256).or_default().append(&mut urls);
    }

    // Create the output zip file
    let (output_zip_file, output_zip_path) =
        tempfile::NamedTempFile::new().unwrap().keep().unwrap();
    let mut writer = zip::ZipWriter::new(output_zip_file);

    // For each file in the manifest
    let manifest = fs.read_to_string("/MANIFEST").await?;
    for line in manifest.lines() {
        if let Some((file_path, sha256)) = line.rsplit_once("=") {
            if !links.urls.contains_key(sha256) {
                // Only files that aren't contained in LINKS
                let data = fs.read(file_path).await?;
                let file_path = file_path.to_owned();
                writer = tokio::task::spawn_blocking(move || {
                    writer
                        .start_file(
                            file_path,
                            zip::write::FileOptions::default()
                                .compression_method(zip::CompressionMethod::Zstd)
                                .large_file(data.len() >= 4 * 1024 * 1024 * 1024),
                        )
                        .unwrap();
                    writer.write_all(&data).unwrap();
                    writer
                })
                .await
                .unwrap();
            }
        } else {
            return Err(CartonError::Other(
                "MANIFEST was not in the form {path}={sha256}",
            ));
        }
    }

    let manifest_data = fs.read("/MANIFEST").await?;
    tokio::task::spawn_blocking(move || {
        // Add MANIFEST
        writer
            .start_file(
                "MANIFEST",
                zip::write::FileOptions::default()
                    .compression_method(zip::CompressionMethod::Stored),
            )
            .unwrap();
        writer.write_all(&manifest_data).unwrap();

        // Add LINKS
        writer
            .start_file(
                "LINKS",
                zip::write::FileOptions::default()
                    .compression_method(zip::CompressionMethod::Stored),
            )
            .unwrap();
        let data = toml::to_vec(&links).unwrap();
        writer.write_all(&data).unwrap();

        // Finish writing the zip file
        log::trace!("Closing zip file writer");
        let mut f = writer.finish().unwrap();
        f.flush().unwrap();
    })
    .await
    .unwrap();

    // Return the output path
    Ok(output_zip_path)
}

#[cfg(test)]
mod tests {
    use super::Links;

    #[test]
    fn test_deserialize_links() {
        let serialized = "
version = 1

[urls]
e550f6224a5133f597d823ab4590f369e0b20e3c6446488225fc6f7a372b9fe2 = [\"https://example.com/file\"]
";

        let deserialized: Links = toml::from_str(serialized).unwrap();
        let target = Links {
            version: 1,
            urls: [(
                "e550f6224a5133f597d823ab4590f369e0b20e3c6446488225fc6f7a372b9fe2".to_owned(),
                ["https://example.com/file".to_owned()].into(),
            )]
            .into_iter()
            .collect(),
        };

        assert_eq!(deserialized, target);
    }
}
