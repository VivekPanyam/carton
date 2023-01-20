//! This module handles parsing a carton.toml file
//! See `docs/specification/format.md` for more details
use std::{collections::HashMap, marker::PhantomData, str::FromStr};

use chrono::{DateTime, Utc};
use serde::{de::Visitor, Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
pub struct CartonToml {
    /// A number defining the carton spec version. Should be 1
    spec_version: u64,

    /// The name of the model
    pub(crate) model_name: Option<String>,

    /// The model description
    pub(crate) model_description: Option<String>,

    /// A list of platforms this model supports
    /// If empty, all platforms are okay
    /// These are target triples
    pub(crate) required_platforms: Option<Vec<Triple>>,

    /// A list of inputs for the model
    /// Can be empty
    pub(crate) input: Option<Vec<TensorSpec>>,

    /// A list of outputs for the model
    /// Can be empty
    pub(crate) output: Option<Vec<TensorSpec>>,

    /// Test data
    /// Can be empty
    pub(crate) self_test: Option<Vec<SelfTest>>,

    /// Examples
    /// Can be empty
    pub(crate) example: Option<Vec<Example>>,

    /// Information about the runner to use
    pub(crate) runner: RunnerInfo,
}

#[derive(Debug, PartialEq)]
pub(crate) struct Triple(pub(crate) target_lexicon::Triple);

impl Serialize for Triple {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.0.to_string().as_ref())
    }
}

impl<'de> Deserialize<'de> for Triple {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        let inner = target_lexicon::Triple::from_str(raw.as_str())
            .map_err(|_| serde::de::Error::custom("invalid target triple"))?;

        Ok(Self(inner))
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
pub struct RunnerInfo {
    pub runner_name: String,
    pub required_framework_version: semver::VersionReq,
    pub runner_compat_version: u64,

    pub opts: Option<HashMap<String, RunnerOpt>>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
#[non_exhaustive]
pub enum RunnerOpt {
    Integer(i64),
    Double(f64),
    String(String),
    Boolean(bool),
    Date(DateTime<Utc>),
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
pub struct SelfTest {
    pub(crate) name: Option<String>,
    pub(crate) description: Option<String>,
    pub(crate) inputs: HashMap<String, TensorReference>,

    // Can be empty
    pub(crate) expected_out: Option<HashMap<String, TensorReference>>,
}

struct RequiredPrefixVisitor<'a, T> {
    req_prefix: &'a str,
    _pd: PhantomData<T>,
}

impl<'a, 'de, T: From<String>> Visitor<'de> for RequiredPrefixVisitor<'a, T> {
    type Value = T;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_fmt(format_args!("Expected prefix of {}", self.req_prefix))
    }

    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        if v.starts_with(self.req_prefix) {
            Ok(v.to_owned().into())
        } else {
            Err(serde::de::Error::custom(format!(
                "expected string with prefix of {}, but got {}",
                self.req_prefix, v
            )))
        }
    }

    fn visit_string<E>(self, v: String) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        self.visit_str(v.as_str())
    }
}

/// References a tensor in @tensor_data
/// Must be a string that starts with `@tensor_data/`
#[derive(Debug, PartialEq)]
pub struct TensorReference(pub(crate) String);

impl Serialize for TensorReference {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for TensorReference {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_string(RequiredPrefixVisitor {
            req_prefix: "@tensor_data/",
            _pd: PhantomData,
        })
    }
}

impl From<String> for TensorReference {
    fn from(value: String) -> Self {
        Self(value)
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
pub struct Example {
    pub(crate) name: Option<String>,
    pub(crate) description: Option<String>,
    pub(crate) inputs: HashMap<String, TensorOrMiscReference>,
    pub(crate) sample_out: HashMap<String, TensorOrMiscReference>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum TensorOrMiscReference {
    T(TensorReference),
    M(MiscFileReference),
}

/// References a file in @misc
/// Must be a string that starts with `@misc/`
#[derive(Debug, PartialEq)]
pub struct MiscFileReference(pub(crate) String);

impl Serialize for MiscFileReference {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for MiscFileReference {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_string(RequiredPrefixVisitor {
            req_prefix: "@misc/",
            _pd: PhantomData,
        })
    }
}

impl From<String> for MiscFileReference {
    fn from(value: String) -> Self {
        Self(value)
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
pub struct TensorSpec {
    pub(crate) name: String,

    /// The datatype
    pub(crate) dtype: DataType,

    /// Tensor shape
    pub(crate) shape: Shape,

    /// Optional description
    pub(crate) description: Option<String>,

    /// Optional internal name
    pub(crate) internal_name: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum DataType {
    Float32,
    Float64,

    String,
    Int8,
    Int16,
    Int32,
    Int64,

    Uint8,
    Uint16,
    Uint32,
    Uint64,
}

#[derive(Debug, PartialEq)]
#[non_exhaustive]
pub enum Shape {
    /// Any shape
    Any,

    /// A symbol for the whole shape
    Symbol(String),

    /// A list of dimensions
    /// An empty vec is considered a scalar
    Shape(Vec<Dimension>),
}

struct ShapeVisitor;

impl<'de> Visitor<'de> for ShapeVisitor {
    type Value = Shape;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("A string or a list")
    }

    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        if v == "*" {
            Ok(Shape::Any)
        } else {
            Ok(Shape::Symbol(v.to_owned()))
        }
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        let mut out = match seq.size_hint() {
            Some(s) => Vec::with_capacity(s),
            None => Vec::new(),
        };

        while let Some(item) = seq.next_element()? {
            out.push(item);
        }

        Ok(Shape::Shape(out))
    }
}

impl<'de> Deserialize<'de> for Shape {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(ShapeVisitor)
    }
}

impl Serialize for Shape {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Shape::Any => serializer.serialize_str("*"),
            Shape::Symbol(s) => serializer.serialize_str(s.as_str()),
            Shape::Shape(s) => s.serialize(serializer),
        }
    }
}

/// A dimension can be either a fixed value, a symbol, or any value
#[derive(Debug, PartialEq)]
#[non_exhaustive]
pub enum Dimension {
    Value(u64),
    Symbol(String),
    Any,
}

struct DimensionVisitor;

impl<'de> Visitor<'de> for DimensionVisitor {
    type Value = Dimension;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("A string or a nonnegative integer")
    }

    fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(Dimension::Value(
            v.try_into().map_err(|_| serde::de::Error::custom("`Dimension`s can only be strings or nonnegative integers, but got a negative integer"))?,
        ))
    }

    fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        if v == "*" {
            Ok(Dimension::Any)
        } else {
            Ok(Dimension::Symbol(v.to_owned()))
        }
    }
}

impl<'de> Deserialize<'de> for Dimension {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(DimensionVisitor)
    }
}

impl Serialize for Dimension {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Dimension::Any => serializer.serialize_str("*"),
            Dimension::Symbol(s) => serializer.serialize_str(s.as_str()),
            Dimension::Value(v) => v.serialize(serializer),
        }
    }
}

pub(crate) async fn parse(data: &[u8]) -> crate::error::Result<CartonToml> {
    Ok(toml::from_slice(data)?)
}

#[cfg(test)]
mod tests {
    use std::{path::PathBuf, str::FromStr};

    use target_lexicon::Triple;

    use crate::format::v1::carton_toml::CartonToml;

    fn get_test_data_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/format/v1/test_data")
    }

    #[test]
    fn test_target_triple() {
        let sample_triples = [
            "aarch64-apple-darwin",
            "x86_64-apple-darwin",
            "aarch64-unknown-linux-gnu",
            "x86_64-unknown-linux-gnu",
        ];

        // Make sure they're all valid
        assert!(sample_triples.iter().all(|t| Triple::from_str(t).is_ok()));

        // Host is the based on the `TARGET` the code was compiled for
        // Make sure it matches one of the above
        assert!(sample_triples
            .iter()
            .any(|t| Triple::host().to_string().as_str() == *t));
    }

    #[test]
    fn parse_all_test_toml_files() {
        // Get all test data files in this dir that end in .toml
        let paths: Vec<_> = get_test_data_dir()
            .read_dir()
            .unwrap()
            .map(|item| item.unwrap().path())
            .filter(|p| p.to_str().unwrap().ends_with(".toml"))
            .collect();

        assert!(!paths.is_empty(), "We expect some test data");

        // Parse them all
        for p in &paths {
            let data = std::fs::read(p).unwrap();

            if p.to_str().unwrap().ends_with("_expect_failure.toml") {
                // Expect error
                assert!(
                    toml::from_slice::<CartonToml>(&data).is_err(),
                    "Expected {:#?} to fail",
                    p
                )
            } else {
                let config: CartonToml = toml::from_slice(&data).unwrap();
                // println!("{:#?}", config);

                // Serializing and deserializing should give us back the original config
                let serialized = toml::to_string_pretty(&config).unwrap();
                let config2: CartonToml = toml::from_str(&serialized).unwrap();

                assert_eq!(config, config2);
            }
        }
    }
}
