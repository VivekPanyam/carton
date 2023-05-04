use std::{
    collections::HashMap,
    hash::Hash,
    pin::Pin,
    str::FromStr,
    sync::{Arc, Mutex},
};

use async_trait::async_trait;
use carton_macros::for_each_carton_type;
use lunchbox::types::{MaybeSend, MaybeSync};
use target_lexicon::Triple;
use tokio::{io::AsyncRead, sync::OnceCell};

use crate::{
    conversion_utils::{ConvertFromWithContext, ConvertIntoWithContext},
    types::{Tensor, TensorStorage},
};

// Info about a carton
pub struct CartonInfo<T>
where
    T: TensorStorage,
{
    /// The name of the model
    pub model_name: Option<String>,

    /// A short description (should be 100 characters or less)
    pub short_description: Option<String>,

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
    pub self_tests: Option<Vec<SelfTest<T>>>,

    /// Examples
    /// Can be empty
    pub examples: Option<Vec<Example<T>>>,

    /// Information about the runner to use
    pub runner: RunnerInfo,

    /// Misc files that can be referenced by the description. The key is a string
    /// starting with `@misc/` followed by a normalized path (i.e one that does not
    /// reference parent directories, etc)
    pub misc_files: Option<HashMap<String, ArcMiscFileLoader>>,
}

impl<T: TensorStorage> Clone for CartonInfo<T> {
    fn clone(&self) -> Self {
        Self {
            model_name: self.model_name.clone(),
            short_description: self.short_description.clone(),
            model_description: self.model_description.clone(),
            required_platforms: self.required_platforms.clone(),
            inputs: self.inputs.clone(),
            outputs: self.outputs.clone(),
            self_tests: self.self_tests.clone(),
            examples: self.examples.clone(),
            runner: self.runner.clone(),
            misc_files: self.misc_files.clone(),
        }
    }
}

/// An internal struct used when loading models. It contains extra things like the
/// manifest hash
pub(crate) struct CartonInfoWithExtras<T>
where
    T: TensorStorage,
{
    pub(crate) info: CartonInfo<T>,

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
        self.get().await;

        // This unwrap is safe because we just ran get above
        self.inner.into_inner().unwrap()
    }
}

impl<T> From<T> for PossiblyLoaded<T> {
    fn from(value: T) -> Self {
        Self::from_value(value)
    }
}

pub struct SelfTest<T>
where
    T: TensorStorage,
{
    pub name: Option<String>,
    pub description: Option<String>,
    pub inputs: HashMap<String, PossiblyLoaded<Tensor<T>>>,

    // Can be empty
    pub expected_out: Option<HashMap<String, PossiblyLoaded<Tensor<T>>>>,
}

impl<T: TensorStorage> Clone for SelfTest<T> {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            description: self.description.clone(),
            inputs: self.inputs.clone(),
            expected_out: self.expected_out.clone(),
        }
    }
}

pub struct Example<T>
where
    T: TensorStorage,
{
    pub name: Option<String>,
    pub description: Option<String>,
    pub inputs: HashMap<String, TensorOrMisc<T>>,
    pub sample_out: HashMap<String, TensorOrMisc<T>>,
}

impl<T: TensorStorage> Clone for Example<T> {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            description: self.description.clone(),
            inputs: self.inputs.clone(),
            sample_out: self.sample_out.clone(),
        }
    }
}

// This isn't ideal, but since it's not on the critical path, it's probably okay
#[cfg(target_family = "wasm")]
pub type MiscFile = Box<dyn AsyncRead + Unpin>;

#[cfg(not(target_family = "wasm"))]
pub type MiscFile = Box<dyn AsyncRead + Send + Sync + Unpin>;

#[cfg_attr(target_family = "wasm", async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait)]
pub trait MiscFileLoader {
    async fn get(&self) -> MiscFile;
}

#[cfg(target_family = "wasm")]
pub type ArcMiscFileLoader = Arc<dyn MiscFileLoader>;

#[cfg(not(target_family = "wasm"))]
pub type ArcMiscFileLoader = Arc<dyn MiscFileLoader + Send + Sync>;

pub enum TensorOrMisc<T>
where
    T: TensorStorage,
{
    Tensor(PossiblyLoaded<Tensor<T>>),
    Misc(ArcMiscFileLoader),
}

impl<T: TensorStorage> Clone for TensorOrMisc<T> {
    fn clone(&self) -> Self {
        match self {
            Self::Tensor(v) => Self::Tensor(v.clone()),
            Self::Misc(v) => Self::Misc(v.clone()),
        }
    }
}

#[derive(Clone)]
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

#[derive(Clone)]
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

#[derive(Clone)]
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
#[derive(Clone)]
pub enum Dimension {
    Value(u64),
    Symbol(String),
    Any,
}

for_each_carton_type! {
    #[derive(Debug, Clone, Copy)]
    pub enum DataType {
        $($CartonType,)*
    }

    impl FromStr for DataType {
        type Err = String;

        fn from_str(s: &str) -> Result<Self, Self::Err> {
            match s {
                $(
                    $TypeStr => Ok(DataType::$CartonType),
                )*
                other => Err(format!("Invalid datatype: `{other}`"))
            }
        }
    }

    impl DataType {
        pub fn to_str(&self) -> &'static str {
            match self {
                $(
                    DataType::$CartonType => $TypeStr,
                )*
            }
        }
    }
}

impl<T, U, C> ConvertFromWithContext<CartonInfoWithExtras<T>, C> for CartonInfoWithExtras<U>
where
    CartonInfo<U>: ConvertFromWithContext<CartonInfo<T>, C>,
    T: TensorStorage,
    U: TensorStorage,
    C: Copy,
{
    fn from(value: CartonInfoWithExtras<T>, context: C) -> Self {
        Self {
            info: value.info.convert_into_with_context(context),
            manifest_sha256: value.manifest_sha256,
        }
    }
}

impl<T, U, C> ConvertFromWithContext<CartonInfo<T>, C> for CartonInfo<U>
where
    SelfTest<U>: ConvertFromWithContext<SelfTest<T>, C>,
    Example<U>: ConvertFromWithContext<Example<T>, C>,
    T: TensorStorage,
    U: TensorStorage,
    C: Copy,
{
    fn from(value: CartonInfo<T>, context: C) -> Self {
        Self {
            model_name: value.model_name,
            short_description: value.short_description,
            model_description: value.model_description,
            required_platforms: value.required_platforms,
            inputs: value.inputs,
            outputs: value.outputs,
            self_tests: value.self_tests.convert_into_with_context(context),
            examples: value.examples.convert_into_with_context(context),
            runner: value.runner,
            misc_files: value.misc_files,
        }
    }
}

impl<T, U, C> ConvertFromWithContext<SelfTest<T>, C> for SelfTest<U>
where
    PossiblyLoaded<Tensor<U>>: ConvertFromWithContext<PossiblyLoaded<Tensor<T>>, C>,
    T: TensorStorage,
    U: TensorStorage,
    C: Copy,
{
    fn from(value: SelfTest<T>, context: C) -> Self {
        Self {
            name: value.name,
            description: value.description,
            inputs: value.inputs.convert_into_with_context(context),
            expected_out: value.expected_out.convert_into_with_context(context),
        }
    }
}

impl<T, U, C> ConvertFromWithContext<Example<T>, C> for Example<U>
where
    PossiblyLoaded<Tensor<U>>: ConvertFromWithContext<PossiblyLoaded<Tensor<T>>, C>,
    T: TensorStorage,
    U: TensorStorage,
    C: Copy,
{
    fn from(value: Example<T>, context: C) -> Self {
        Self {
            name: value.name,
            description: value.description,
            inputs: value.inputs.convert_into_with_context(context),
            sample_out: value.sample_out.convert_into_with_context(context),
        }
    }
}

impl<T, U, C> ConvertFromWithContext<TensorOrMisc<T>, C> for TensorOrMisc<U>
where
    PossiblyLoaded<Tensor<U>>: ConvertFromWithContext<PossiblyLoaded<Tensor<T>>, C>,
    T: TensorStorage,
    U: TensorStorage,
    C: Copy,
{
    fn from(value: TensorOrMisc<T>, context: C) -> Self {
        match value {
            TensorOrMisc::Tensor(t) => Self::Tensor(t.convert_into_with_context(context)),
            TensorOrMisc::Misc(m) => Self::Misc(m),
        }
    }
}

impl<T, U, C> ConvertFromWithContext<PossiblyLoaded<T>, C> for PossiblyLoaded<U>
where
    U: ConvertFromWithContext<T, C> + MaybeSend,
    T: MaybeSync + MaybeSend + 'static,
    C: MaybeSend + 'static,
    C: Copy,
{
    fn from(value: PossiblyLoaded<T>, context: C) -> Self {
        Self::from_loader(Box::pin(async move {
            value
                .into_get()
                .await
                .unwrap()
                .convert_into_with_context(context)
        }))
    }
}

impl<T, U, C> ConvertFromWithContext<Option<T>, C> for Option<U>
where
    U: ConvertFromWithContext<T, C>,
    C: Copy,
{
    fn from(value: Option<T>, context: C) -> Self {
        value.map(|item| item.convert_into_with_context(context))
    }
}

impl<T, U, C> ConvertFromWithContext<Vec<T>, C> for Vec<U>
where
    U: ConvertFromWithContext<T, C>,
    C: Copy,
{
    fn from(value: Vec<T>, context: C) -> Self {
        value
            .into_iter()
            .map(|item| item.convert_into_with_context(context))
            .collect()
    }
}

impl<K, T, U, C> ConvertFromWithContext<HashMap<K, T>, C> for HashMap<K, U>
where
    U: ConvertFromWithContext<T, C>,
    K: Hash + Eq,
    C: Copy,
{
    fn from(value: HashMap<K, T>, context: C) -> Self {
        value
            .into_iter()
            .map(|(k, item)| (k, item.convert_into_with_context(context)))
            .collect()
    }
}
