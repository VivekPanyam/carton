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

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};

use path_clean::PathClean;
use runner_interface_v1::slowlog::slowlog;
use sha2::{Digest, Sha256};
use tempfile::TempDir;
use walkdir::WalkDir;

use crate::conversion_utils::{convert_opt_map, convert_opt_vec, convert_vec};
use crate::error::{CartonError, Result};
use crate::format::v1::links::Links;
use crate::types::PackOpts;

use super::carton_toml::{CartonToml, TensorOrMiscReference};

// Util to save a misc file
async fn save_misc_file<'a>(
    misc_dir: &'a std::path::Path,
    name: &'a str,
    item: crate::info::ArcMiscFileLoader,
) -> Result<()> {
    // TODO: verify that name is a normalized path
    let fname = name;
    let mut file = tokio::fs::File::create(misc_dir.join(fname)).await?;
    let mut reader = item.get().await;
    tokio::io::copy(reader.as_mut(), &mut file).await?;

    Ok(())
}

/// Given a path to a filled `model` dir, this function creates a complete carton by saving all the additonal
/// info. Returns a path to the saved file
pub(crate) async fn save(
    pack_opts: PackOpts,
    model_dir_path: &std::path::Path,
) -> Result<std::path::PathBuf> {
    // Extract the model info from pack opts
    let info = pack_opts.info;

    // Extract info about linked files if any
    let linked_files: Option<Links> = pack_opts.linked_files.map(|v| v.into());

    // Create a tempdir
    let tempdir = TempDir::new().unwrap();

    // Check that info.short_description is <= 100 characters
    if let Some(desc) = &info.short_description {
        if desc.len() > 100 {
            // More than 100 bytes. Check if it's also more than 100 chars
            if desc.chars().count() > 100 {
                return Err(CartonError::Other(
                    "The provided short_description is > 100 chars long.",
                ));
            }
        }
    }

    // Create the carton.toml we're going to write out
    let mut config = CartonToml {
        spec_version: 1, // Format V1
        model_name: info.model_name,
        short_description: info.short_description,
        model_description: info.model_description,
        license: info.license,
        repository: info.repository,
        homepage: info.homepage,
        required_platforms: convert_opt_vec(info.required_platforms),
        input: convert_opt_vec(info.inputs),
        output: convert_opt_vec(info.outputs),
        self_test: None,
        example: None,
        runner: info.runner.into(),
    };

    // 1. Save all the misc files
    log::trace!("Processing misc files...");
    let misc_dir = tempdir.path().join("misc");
    tokio::fs::create_dir(&misc_dir).await?;
    let mut misc_file_counter = 0;

    // 1. Save all the misc files
    // TODO: handle misc files in the examples
    if let Some(misc_files) = info.misc_files {
        for (name, item) in misc_files {
            save_misc_file(&misc_dir, &name, item).await.unwrap();
        }
    }

    // 2. Save all the tensors
    log::trace!("Processing examples and self tests...");
    let mut tensors_to_save = HashMap::new();
    let mut counter = 0;

    // TODO: Future optimization: if we see the same tensor multiple times, write it out once
    if let Some(self_tests) = info.self_tests {
        let mut out_self_tests = Vec::new();
        for item in self_tests {
            let mut out_inputs = HashMap::new();
            let mut out_expected_out = None;

            // Save the inputs
            for (k, v) in item.inputs {
                let save_key = format!("@tensor_data/_tensor_{counter}");
                tensors_to_save.insert(save_key.clone(), v);
                out_inputs.insert(k, save_key.into());
                counter += 1;
            }

            // Save the expected outputs (if any)
            if let Some(expected_out) = item.expected_out {
                let mut to_output = HashMap::new();
                for (k, v) in expected_out {
                    let save_key = format!("@tensor_data/_tensor_{counter}");
                    tensors_to_save.insert(save_key.clone(), v);
                    to_output.insert(k, save_key.into());
                    counter += 1;
                }

                out_expected_out = Some(to_output);
            }

            out_self_tests.push(super::carton_toml::SelfTest {
                name: item.name,
                description: item.description,
                inputs: out_inputs,
                expected_out: out_expected_out,
            });
        }

        config.self_test = Some(out_self_tests);
    }

    if let Some(examples) = info.examples {
        let mut out_examples = Vec::new();
        for item in examples {
            let mut out_inputs = HashMap::new();
            let mut out_sample_out = HashMap::new();

            // Save the inputs
            for (k, v) in item.inputs {
                match v {
                    crate::info::TensorOrMisc::Tensor(t) => {
                        let save_key = format!("@tensor_data/_tensor_{counter}");
                        tensors_to_save.insert(save_key.clone(), t);
                        out_inputs.insert(k, TensorOrMiscReference::T(save_key.into()));
                        counter += 1;
                    }
                    crate::info::TensorOrMisc::Misc(m) => {
                        let save_key = format!("@misc/_example_misc_file_{misc_file_counter}");
                        save_misc_file(
                            &misc_dir,
                            &format!("_example_misc_file_{misc_file_counter}"),
                            m,
                        )
                        .await
                        .unwrap();
                        out_inputs.insert(k, TensorOrMiscReference::M(save_key.into()));
                        misc_file_counter += 1;
                    }
                }
            }

            // Save the outputs
            for (k, v) in item.sample_out {
                match v {
                    crate::info::TensorOrMisc::Tensor(t) => {
                        let save_key = format!("@tensor_data/_tensor_{counter}");
                        tensors_to_save.insert(save_key.clone(), t);
                        out_sample_out.insert(k, TensorOrMiscReference::T(save_key.into()));
                        counter += 1;
                    }
                    crate::info::TensorOrMisc::Misc(m) => {
                        let save_key = format!("@misc/_example_misc_file_{misc_file_counter}");
                        save_misc_file(
                            &misc_dir,
                            &format!("_example_misc_file_{misc_file_counter}"),
                            m,
                        )
                        .await
                        .unwrap();
                        out_sample_out.insert(k, TensorOrMiscReference::M(save_key.into()));
                        misc_file_counter += 1;
                    }
                }
            }

            out_examples.push(super::carton_toml::Example {
                name: item.name,
                description: item.description,
                inputs: out_inputs,
                sample_out: out_sample_out,
            })
        }

        config.example = Some(out_examples);
    }

    // Load all the tensors
    let mut loaded = HashMap::new();
    for k in tensors_to_save.keys() {
        loaded.insert(k.clone(), tensors_to_save.get(k).unwrap().get().await);
    }

    // Save them
    let tensor_data_dir = tempdir.path().join("tensor_data");
    tokio::fs::create_dir(&tensor_data_dir).await?;
    super::tensor::save_tensors(&tensor_data_dir, loaded).unwrap();

    // 3. Generate a carton.toml file
    log::trace!("Writing carton.toml");
    let serialized = toml::to_string_pretty(&config).unwrap();
    tokio::fs::write(tempdir.path().join("carton.toml"), serialized)
        .await
        .unwrap();

    // 4. Zip up all the files and folders
    log::trace!("Creating ZipFileWriter");
    let (output_zip_file, output_zip_path) =
        tempfile::NamedTempFile::new().unwrap().keep().unwrap();
    let mut writer = zip::ZipWriter::new(output_zip_file);

    // Generate a MANIFEST as we're zipping files and folders
    log::trace!("Packing metadata");
    let mut manifest_contents = BTreeMap::new();
    let mut symlink_targets = HashMap::new();
    for entry in WalkDir::new(&tempdir) {
        let entry = entry.unwrap();
        if entry.file_type().is_dir() {
            continue;
        }

        let relative_path = entry
            .path()
            .strip_prefix(&tempdir)
            .unwrap()
            .to_str()
            .unwrap()
            .to_owned();

        // Load the data and compute the sha256
        let mut hasher = Sha256::new();
        let data = tokio::fs::read(entry.path()).await.unwrap();
        hasher.update(&data);
        let sha256 = format!("{:x}", hasher.finalize());
        manifest_contents.insert(relative_path.clone(), Some(sha256));

        // Add the entry to the zip file
        writer = tokio::task::spawn_blocking(move || {
            writer
                .start_file(
                    relative_path,
                    zip::write::FileOptions::default()
                        .compression_method(zip::CompressionMethod::Zstd),
                )
                .unwrap();
            writer.write_all(&data).unwrap();
            writer
        })
        .await
        .unwrap();
    }

    // Add the model dir
    log::trace!("Packing model dir");
    for entry in WalkDir::new(&model_dir_path).follow_links(true) {
        let entry = entry.unwrap();
        if entry.file_type().is_dir() {
            continue;
        }

        let relative_path = Path::new("model")
            .join(entry.path().strip_prefix(&model_dir_path).unwrap())
            .to_str()
            .unwrap()
            .to_owned();

        log::trace!("About to pack {}", &relative_path);
        let mut sl = slowlog(format!("Packaging file '{}'", &relative_path), 5)
            .await
            .without_progress();

        // Should we store this file as a symlink?
        let symlink_target = if entry.path_is_symlink() {
            let absolute_file_path = entry.path();
            assert!(absolute_file_path.is_absolute());

            // Get the target
            let symlink_target = tokio::fs::read_link(absolute_file_path).await.unwrap();

            // Make the target absolute
            let symlink_target = if symlink_target.is_relative() {
                absolute_file_path.parent().unwrap().join(symlink_target)
            } else {
                symlink_target
            };

            // Normalize the path
            let symlink_target = symlink_target.clean();

            // Decide what to do
            if symlink_target.starts_with(&model_dir_path) {
                // Store as a relative symlink
                Some(
                    pathdiff::diff_paths(symlink_target, absolute_file_path.parent().unwrap())
                        .unwrap(),
                )
            } else {
                // The symlink points outside the model dir; store as a file
                None
            }
        } else {
            // Not a symlink
            None
        };

        // Handle symlinks
        if let Some(symlink_target) = symlink_target {
            // Turn it into a string
            let symlink_target = symlink_target.to_str().unwrap().to_owned();

            // Store an empty sha256 for now and we'll update it after all the files have been added
            manifest_contents.insert(relative_path.clone(), None);

            // Store the symlink target for us to use later
            symlink_targets.insert(relative_path.clone(), symlink_target.clone());

            writer
                .add_symlink(
                    relative_path,
                    symlink_target,
                    zip::write::FileOptions::default(),
                )
                .unwrap();
        } else {
            // Load the data and compute the sha256
            let mut hasher = Sha256::new();
            let data = tokio::fs::read(entry.path()).await.unwrap();

            log::trace!("Done reading file {}", &relative_path);

            let (data, sha256) = tokio::task::spawn_blocking(move || {
                hasher.update(&data);
                (data, format!("{:x}", hasher.finalize()))
            })
            .await
            .unwrap();

            log::trace!("Computed sha256 of {}", &relative_path);

            // Only store the file in the zip if (1) we don't have any linked files or (2) the linked files don't include this sha256
            if linked_files
                .as_ref()
                .map_or(true, |v| !v.urls.contains_key(&sha256))
            {
                // Add the entry to the zip file
                let relative_path = relative_path.clone();
                writer = tokio::task::spawn_blocking(move || {
                    writer
                        .start_file(
                            relative_path,
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

            manifest_contents.insert(relative_path, Some(sha256));
        }

        log::trace!("Wrote to zip file");

        sl.done();
    }

    // Get sha256 values for all the symlinks
    let manifest_contents = manifest_contents
        .iter()
        .map(|(k, v)| {
            if v.is_none() {
                let mut path = k.clone();
                let mut visited = HashSet::new();
                loop {
                    let target = symlink_targets.get(&path).unwrap();

                    // `target` is a relative path so we need to convert it to absolute
                    let target = PathBuf::from(path).parent().unwrap().join(target);

                    // Normalize the target
                    let target = target.clean().to_str().unwrap().to_owned();

                    let sha = manifest_contents.get(&target).unwrap();

                    if visited.contains(&target) {
                        // We've already seen this
                        // TODO: don't panic
                        panic!("Got symlink loop when packing a model! File: {k}");
                    }

                    visited.insert(target.clone());

                    match sha {
                        None => {
                            // A symlink to a symlink so we should keep looping
                            path = target;
                        }

                        Some(sha) => {
                            // Got the target
                            return (k, sha.clone());
                        }
                    }
                }
            }

            (k, v.as_ref().unwrap().clone())
        })
        .collect::<BTreeMap<_, _>>();

    // 5. Write the manifest to the zip file in alphabetical order (we're using a BTreeMap for manifest_contents)
    log::trace!("Writing manifest");
    let mut manifest_str = String::new();
    for (k, v) in manifest_contents {
        manifest_str += &format!("{k}={v}\n");
    }

    tokio::task::spawn_blocking(move || {
        writer
            .start_file(
                "MANIFEST",
                zip::write::FileOptions::default()
                    .compression_method(zip::CompressionMethod::Stored),
            )
            .unwrap();
        writer.write_all(manifest_str.as_bytes()).unwrap();

        // 6. Add links (if any)
        if let Some(linked_files) = linked_files {
            // Add LINKS
            writer
                .start_file(
                    "LINKS",
                    zip::write::FileOptions::default()
                        .compression_method(zip::CompressionMethod::Stored),
                )
                .unwrap();
            let data = toml::to_vec(&linked_files).unwrap();
            writer.write_all(&data).unwrap();
        }

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

impl From<target_lexicon::Triple> for super::carton_toml::Triple {
    fn from(value: target_lexicon::Triple) -> Self {
        Self(value)
    }
}

impl From<crate::info::TensorSpec> for super::carton_toml::TensorSpec {
    fn from(value: crate::info::TensorSpec) -> Self {
        Self {
            name: value.name,
            dtype: value.dtype.into(),
            shape: value.shape.into(),
            description: value.description,
            internal_name: value.internal_name,
        }
    }
}

impl From<crate::info::DataType> for super::carton_toml::DataType {
    fn from(value: crate::info::DataType) -> Self {
        match value {
            crate::info::DataType::Float => Self::Float32,
            crate::info::DataType::Double => Self::Float64,
            crate::info::DataType::String => Self::String,
            crate::info::DataType::I8 => Self::Int8,
            crate::info::DataType::I16 => Self::Int16,
            crate::info::DataType::I32 => Self::Int32,
            crate::info::DataType::I64 => Self::Int64,
            crate::info::DataType::U8 => Self::Uint8,
            crate::info::DataType::U16 => Self::Uint16,
            crate::info::DataType::U32 => Self::Uint32,
            crate::info::DataType::U64 => Self::Uint64,
        }
    }
}

impl From<crate::info::Shape> for super::carton_toml::Shape {
    fn from(value: crate::info::Shape) -> Self {
        match value {
            crate::info::Shape::Any => Self::Any,
            crate::info::Shape::Symbol(v) => Self::Symbol(v),
            crate::info::Shape::Shape(v) => Self::Shape(convert_vec(v)),
        }
    }
}

impl From<crate::info::Dimension> for super::carton_toml::Dimension {
    fn from(value: crate::info::Dimension) -> Self {
        match value {
            crate::info::Dimension::Value(v) => Self::Value(v),
            crate::info::Dimension::Symbol(v) => Self::Symbol(v),
            crate::info::Dimension::Any => Self::Any,
        }
    }
}

impl From<crate::info::RunnerInfo> for super::carton_toml::RunnerInfo {
    fn from(value: crate::info::RunnerInfo) -> Self {
        Self {
            runner_name: value.runner_name,
            required_framework_version: value.required_framework_version,
            runner_compat_version: value
                .runner_compat_version
                .expect("runner_compat_version should be set by the time `save` is called"),
            opts: convert_opt_map(value.opts),
        }
    }
}

impl From<crate::info::RunnerOpt> for super::carton_toml::RunnerOpt {
    fn from(value: crate::info::RunnerOpt) -> Self {
        match value {
            crate::info::RunnerOpt::Integer(v) => Self::Integer(v),
            crate::info::RunnerOpt::Double(v) => Self::Double(v),
            crate::info::RunnerOpt::String(v) => Self::String(v),
            crate::info::RunnerOpt::Boolean(v) => Self::Boolean(v),
        }
    }
}
