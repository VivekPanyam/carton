use std::{
    collections::HashMap,
    pin::Pin,
    sync::{Arc, Mutex},
};

use carton_macros::for_each_carton_type;
use target_lexicon::Triple;
use tokio::{io::AsyncRead, sync::OnceCell};

use crate::types::Tensor;

// Info about a carton
pub struct CartonInfo {
    /// The name of the model
    pub model_name: Option<String>,

    /// The model description
    pub model_description: Option<String>,

    /// A list of platforms this model supports
    /// If empty or unspecified, all platforms are okay
    pub required_platforms: Option<Vec<Triple>>,

    /// A list of inputs for the model
    /// Can be empty
    pub inputs: Option<Vec<TensorSpec>>,

    /// A list of outputs for the model
    /// Can be empty
    pub outputs: Option<Vec<TensorSpec>>,

    /// Test data
    /// Can be empty
    pub self_tests: Option<Vec<SelfTest>>,

    /// Examples
    /// Can be empty
    pub examples: Option<Vec<Example>>,

    /// Information about the runner to use
    pub runner: RunnerInfo,

    /// Misc files that can be referenced by the description. The key is a string
    /// starting with `@misc/` followed by a normalized path (i.e one that does not
    /// reference parent directories, etc)
    pub misc_files: Option<HashMap<String, PossiblyLoaded<MiscFile>>>,
}

/// An internal struct used when loading models. It contains extra things like the
/// manifest hash
pub(crate) struct CartonInfoWithExtras {
    pub(crate) info: CartonInfo,

    /// The sha256 of the MANIFEST file (if available)
    /// This should always be available unless we're running an unpacked model
    pub(crate) manifest_sha256: Option<String>,
}

#[cfg(target_family = "wasm")]
pub type BoxFuture<'a, T> = Pin<Box<dyn std::future::Future<Output = T> + 'a>>;

#[cfg(not(target_family = "wasm"))]
pub type BoxFuture<'a, T> = Pin<Box<dyn std::future::Future<Output = T> + Send + 'a>>;

/// Something that is possibly loaded
pub struct PossiblyLoaded<T> {
    inner: Arc<PossiblyLoadedInner<T>>,
}

impl<T> Clone for PossiblyLoaded<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T> PossiblyLoaded<T> {
    pub fn from_value(value: T) -> Self {
        Self {
            inner: Arc::new(PossiblyLoadedInner {
                inner: OnceCell::new_with(Some(value)),
                loader: Default::default(),
            }),
        }
    }

    pub fn from_loader(loader: BoxFuture<'static, T>) -> Self {
        Self {
            inner: Arc::new(PossiblyLoadedInner {
                inner: Default::default(),
                loader: Mutex::new(Some(loader)),
            }),
        }
    }

    pub async fn get(&self) -> &T {
        self.inner.get().await
    }

    pub async fn into_get(self) -> Option<T> {
        let inner = Arc::try_unwrap(self.inner);
        match inner {
            Ok(inner) => Some(inner.into_inner().await),
            Err(_) => None,
        }
    }
}

struct PossiblyLoadedInner<T> {
    inner: OnceCell<T>,

    // This type is kinda messy so that `PossiblyLoaded` implements Sync
    loader: Mutex<Option<BoxFuture<'static, T>>>,
}

impl<T> PossiblyLoadedInner<T> {
    async fn get(&self) -> &T {
        match self.inner.get() {
            Some(value) => value,
            None => {
                // We need to initialize
                let loader = { self.loader.lock().unwrap().take() };
                self.inner
                    .get_or_init(|| async move { loader.unwrap().await })
                    .await
            }
        }
    }

    async fn into_inner(self) -> T {
        // Run `get` to ensure we load the value
        self.get();

        // This unwrap is safe because we just ran get above
        self.inner.into_inner().unwrap()
    }
}

impl<T> From<T> for PossiblyLoaded<T> {
    fn from(value: T) -> Self {
        Self::from_value(value)
    }
}

pub struct SelfTest {
    pub name: Option<String>,
    pub description: Option<String>,
    pub inputs: HashMap<String, PossiblyLoaded<Tensor>>,

    // Can be empty
    pub expected_out: Option<HashMap<String, PossiblyLoaded<Tensor>>>,
}

pub struct Example {
    pub name: Option<String>,
    pub description: Option<String>,
    pub inputs: HashMap<String, TensorOrMisc>,
    pub sample_out: HashMap<String, TensorOrMisc>,
}

// This isn't ideal, but since it's not on the critical path, it's probably okay
#[cfg(target_family = "wasm")]
pub type MiscFile = Box<dyn AsyncRead>;

#[cfg(not(target_family = "wasm"))]
pub type MiscFile = Box<dyn AsyncRead + Send + Sync>;

pub enum TensorOrMisc {
    Tensor(PossiblyLoaded<Tensor>),
    Misc(PossiblyLoaded<MiscFile>),
}

pub struct RunnerInfo {
    /// The name of the runner to use
    pub runner_name: String,

    /// The required framework version range to run the model with
    /// This is a semver version range. See https://docs.rs/semver/1.0.16/semver/struct.VersionReq.html
    /// for format details.
    /// For example `=1.2.4`, means exactly version `1.2.4`
    /// In most cases, this should be exactly one version
    pub required_framework_version: semver::VersionReq,

    /// Don't set this unless you know what you're doing
    pub runner_compat_version: Option<u64>,

    /// Options to pass to the runner. These are runner-specific (e.g.
    /// PyTorch, TensorFlow, etc).
    ///
    /// Sometimes used to configure thread-pool sizes, etc.
    /// See the documentation for more info
    pub opts: Option<HashMap<String, RunnerOpt>>,
}

/// The types of options that can be passed to runners
#[derive(Clone)]
pub enum RunnerOpt {
    Integer(i64),
    Double(f64),
    String(String),
    Boolean(bool),
}

#[non_exhaustive]
pub struct TensorSpec {
    pub name: String,

    /// The datatype
    pub dtype: DataType,

    /// Tensor shape
    pub shape: Shape,

    /// Optional description
    pub description: Option<String>,

    /// Optional internal name
    pub internal_name: Option<String>,
}

pub enum Shape {
    /// Any shape
    Any,

    /// A symbol for the whole shape
    Symbol(String),

    /// A list of dimensions
    /// An empty vec is considered a scalar
    Shape(Vec<Dimension>),
}

/// A dimension can be either a fixed value, a symbol, or any value
pub enum Dimension {
    Value(u64),
    Symbol(String),
    Any,
}

for_each_carton_type! {
    #[derive(Debug)]
    pub enum DataType {
        $($CartonType,)*
    }
}
