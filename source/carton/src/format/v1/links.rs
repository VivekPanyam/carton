use std::collections::HashMap;

use async_zip::{write::ZipFileWriter, ZipEntryBuilder};
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

/// Take a path to a packed carton along with a map from sha256 to urls and shrink the carton by storing
/// URLs instead of the orig files when possible
#[cfg(not(target_family = "wasm"))]
pub(crate) async fn create_links(
    path: std::path::PathBuf,
    urls: HashMap<String, Vec<String>>,
) -> crate::error::Result<std::path::PathBuf> {
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
    let mut output_zip_file = tokio::fs::File::from_std(output_zip_file);
    let mut writer = ZipFileWriter::new(&mut output_zip_file);

    // For each file in the manifest
    let manifest = fs.read_to_string("/MANIFEST").await?;
    for line in manifest.lines() {
        if let Some((file_path, sha256)) = line.rsplit_once("=") {
            if !links.urls.contains_key(sha256) {
                // Only files that aren't contained in LINKS
                let zip_entry =
                    ZipEntryBuilder::new(file_path.into(), async_zip::Compression::Zstd);
                let data = fs.read(file_path).await?;
                writer.write_entry_whole(zip_entry, &data).await.unwrap();
            }
        } else {
            return Err(CartonError::Other(
                "MANIFEST was not in the form {path}={sha256}",
            ));
        }
    }

    // Add MANIFEST
    let zip_entry = ZipEntryBuilder::new("/MANIFEST".into(), async_zip::Compression::Zstd);
    let data = fs.read("/MANIFEST").await?;
    writer.write_entry_whole(zip_entry, &data).await.unwrap();

    // Add LINKS
    let zip_entry = ZipEntryBuilder::new("/LINKS".into(), async_zip::Compression::Zstd);
    let data = toml::to_vec(&links).unwrap();
    writer.write_entry_whole(zip_entry, &data).await.unwrap();

    // Finish writing the zip file
    writer.close().await.unwrap();

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
