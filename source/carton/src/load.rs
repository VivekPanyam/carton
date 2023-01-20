//! This module handles loading a carton

use std::sync::Arc;

use async_trait::async_trait;
use lazy_static::lazy_static;
use lunchbox::{
    path::{LunchboxPathUtils, PathBuf},
    types::{MaybeSend, MaybeSync},
};
use semver::VersionReq;
use url::{ParseError, Url};
use zipfs::{GetReader, ZipFS};

use crate::{
    error::CartonError,
    http::HTTPFile,
    types::{CartonInfo, Device, LoadOpts}, info::CartonInfoWithExtras,
};

/// Load a carton given a url or path and options
pub(crate) async fn load(url_or_path: &str, opts: LoadOpts) -> ReturnType {
    // There are 5 steps to loading a carton:
    // 1. Fetch: Get the file or directory
    // 2. Unwrap the container if any (currently only zip files)
    // 3. Resolve links if necessary
    // 4. Load carton info from the resolved FS
    // 5. Figure out what runner to use (or get it if necessary) and launch the runner
    //
    // Because the output type of each step generally can't be known ahead of time, this
    // process is implemented in a slightly odd way. Step 1 calls into step 2 which calls into step 3
    // and so on. The final step returns a value (of a type that is known ahead of time).
    // This simplifies types and avoids dynamic dispatch (at the cost of a larger binary because of
    // monomorphization).
    fetch(url_or_path, opts, false).await
}

pub(crate) async fn get_carton_info(url_or_path: &str) -> crate::error::Result<CartonInfoWithExtras> {
    let (info, _) = fetch(url_or_path, Default::default(), true).await?;
    Ok(info)
}

/// The return type of `load`
pub(crate) type ReturnType = crate::error::Result<(CartonInfoWithExtras, Option<Runner>)>;

/// All the versions of the runner interface that we support
pub(crate) enum Runner {
    V1(runner_interface_v1::Runner),
}

/// The maximum version of the runner interface supported by this build of carton
const MAX_SUPPORTED_INTERFACE_VERSION: u64 = 1;

/// Step 1: Fetch the file or directory (and call into step 2)
/// If `url` points to a dir on disk, load a local lunchbox filesystem and
/// call directly into step 3
/// If `skip_runner` is true, a runner will not be launched. Only CartonInfo will be returned.
async fn fetch(url: &str, opts: LoadOpts, skip_runner: bool) -> ReturnType {
    let url = parse_protocol(url);
    match url {
        #[cfg(not(target_family = "wasm"))]
        LocatorWithProtocol::LocalFilePath(path) => {
            if tokio::fs::metadata(&path.0).await?.is_dir() {
                // This is a local directory (or a symlink to one)
                // Skip directly to step 3
                maybe_resolve_links(&Arc::new(lunchbox::LocalFS::with_base_dir(path.0).unwrap()), opts, skip_runner).await
            } else {
                // This is a file (or a symlink to one)
                unwrap_container(path, opts, skip_runner).await
            }
        }
        #[cfg(target_family = "wasm")]
        LocatorWithProtocol::LocalFilePath(_) => panic!("Local file paths not supported on wasm!"),
        LocatorWithProtocol::HttpURL(url) => unwrap_container(url, opts, skip_runner).await,
    }
}

/// Optional Step 2: Unwrap a container (e.g. zip) (and call into step 3)
async fn unwrap_container<T>(item: T, opts: LoadOpts, skip_runner: bool) -> ReturnType
where
    T: GetReader + 'static + MaybeSync + MaybeSend,
    T::R: MaybeSync + MaybeSend,
{
    // We currently only support zip so there isn't a whole lot to do here
    let zip = ZipFS::new(item).await;

    maybe_resolve_links(&Arc::new(zip), opts, skip_runner).await
}

/// Step 3: Resolve links (and call into step 4)
async fn maybe_resolve_links<T>(fs: &Arc<T>, opts: LoadOpts, skip_runner: bool) -> ReturnType
where
    T: lunchbox::ReadableFileSystem + MaybeSend + MaybeSync + 'static,
    T::FileType: lunchbox::types::ReadableFile + MaybeSend + MaybeSync + Unpin,
{
    // Basically an overlay filesystem using the `LINKS` file and `MANIFEST` to decide where
    // to direct operations (if necessary)
    let has_manifest = PathBuf::from("/MANIFEST").exists(fs.as_ref()).await;
    let has_links = PathBuf::from("/LINKS").exists(fs.as_ref()).await;

    if !has_manifest {
        // Not a valid carton
        // Return an error
        todo!()
    }

    if !has_links {
        // No links to resolve so just pass through
        load_carton(fs, opts, skip_runner).await
    } else {
        // Resolve links and then make an overlayfs and
        // pass through to load_carton

        todo!()
    }
}

/// Step 4: Load carton info from the resolved fs (and call into step 5)
async fn load_carton<T>(fs: &Arc<T>, opts: LoadOpts, skip_runner: bool) -> ReturnType
where
    T: lunchbox::ReadableFileSystem + MaybeSend + MaybeSync + 'static,
    T::FileType: lunchbox::types::ReadableFile + MaybeSend + MaybeSync + Unpin,
{
    // First, figure out which format version this is
    // Currently, there's only one so we always pass through to it
    let mut info_with_extras = crate::format::v1::load(fs).await?;

    // Merge in load opts
    if let Some(v) = opts.override_runner_name {
        info_with_extras.info.runner.runner_name = v;
    }

    if let Some(v) = opts.override_required_framework_version {
        info_with_extras.info.runner.required_framework_version = VersionReq::parse(&v).map_err(|_| {
            CartonError::Other(
                "`override_required_framework_version` was not a valid semver version range",
            )
        })?;
    }

    if let Some(v) = opts.override_runner_opts {
        info_with_extras.info.runner.opts = if let Some(mut orig) = info_with_extras.info.runner.opts {
            for (k, val) in v.into_iter() {
                orig.insert(k, val);
            }

            Some(orig)
        } else {
            Some(v)
        }
    }

    if skip_runner {
        Ok((info_with_extras, None))
    } else {
        let runner = discover_or_get_runner_and_launch(fs, &info_with_extras, opts.visible_device).await?;

        Ok((info_with_extras, Some(runner)))
    }
}

// Step 5: Figure out what runner to use (or get it if necessary) and launch the runner
#[cfg(not(target_family = "wasm"))]
async fn discover_or_get_runner_and_launch<T>(
    fs: &Arc<T>,
    c: &CartonInfoWithExtras,
    visible_device: Device,
) -> crate::error::Result<Runner>
where
    T: lunchbox::ReadableFileSystem + MaybeSend + MaybeSync + 'static,
    T::FileType: lunchbox::types::ReadableFile + MaybeSend + MaybeSync + Unpin,
{
    // TODO: maybe we want to just do this once at startup or cache it?
    let local_runners = crate::discovery::discover_runners().await;

    // Filter the runners to ones that match our requirements
    let candidate = local_runners
        .into_iter()
        .filter(|runner| {
            // The runner name must be the same as the model we're trying to load
            (runner.runner_name == c.info.runner.runner_name)

            // The runner compat version must be the same as the model we're trying to load
            // (this is kind of like a version for the `model` directory)
            && (runner.runner_compat_version == c.info.runner.runner_compat_version)

            // The runner's framework_version must satisfy the model's required range
            && (c
                .info
                .runner
                .required_framework_version
                .matches(&runner.framework_version))

            // Finally, we must be able to communicate with the runner (so it's interface
            // version should be one we support)
            && (runner.runner_interface_version <= MAX_SUPPORTED_INTERFACE_VERSION)
        })
        // Pick the newest one that matches the requirements
        .max_by_key(|item| item.runner_release_date);

    if let Some(candidate) = candidate {
        // We have a runner we can use!

        match candidate.runner_interface_version {
            // Find the right interface to use
            1 => {
                let runner = runner_interface_v1::Runner::new(
                    &std::path::PathBuf::from(candidate.runner_path),
                    visible_device.clone().into(),
                ).await.unwrap();

                runner.load(
                    fs,
                    c.info.runner.runner_name.clone(),
                    c.info.runner.required_framework_version.clone(),
                    c.info.runner.runner_compat_version,
                    c.info.runner.opts.clone().map(|item| item.into_iter().map(|(k,v)| (k, v.into())).collect()),
                    visible_device.into(),
                    c.manifest_sha256.clone(),
                ).await.map_err(|e| CartonError::ErrorFromRunner(e))?;

                Ok(Runner::V1(runner))

            },
            version => unreachable!("This runner requires a newer interface ({version}) than we have. Shouldn't happen because of the check above."),
        }
    } else {
        // We need to fetch a runner and retry

        todo!()
    }
}

impl From<Device> for runner_interface_v1::types::Device {
    fn from(value: Device) -> Self {
        match value {
            Device::CPU => Self::CPU,
            Device::GPU { uuid } => Self::GPU { uuid },
        }
    }
}

impl From<crate::types::RunnerOpt> for runner_interface_v1::types::RunnerOpt {
    fn from(value: crate::types::RunnerOpt) -> Self {
        match value {
            crate::info::RunnerOpt::Integer(v) => Self::Integer(v),
            crate::info::RunnerOpt::Double(v) => Self::Double(v),
            crate::info::RunnerOpt::String(v) => Self::String(v),
            crate::info::RunnerOpt::Boolean(v) => Self::Boolean(v),
            crate::info::RunnerOpt::Date(v) => Self::Date(v),
        }
    }
}

// No discovery for wasm - just launch a runner and return
#[cfg(target_family = "wasm")]
async fn discover_or_get_runner_and_launch<T>(
    fs: &Arc<T>,
    c: &CartonInfoWithExtras,
    visible_device: Device,
) -> crate::error::Result<Runner>
where
    T: lunchbox::ReadableFileSystem,
{
    todo!()
}

/// Given a url or a path, figure out what protocol it's using
fn parse_protocol(input: &str) -> LocatorWithProtocol {
    match Url::parse(input) {
        Ok(parsed) => match parsed.scheme() {
            "file" => LocatorWithProtocol::LocalFilePath(input.into()),
            "http" | "https" => LocatorWithProtocol::HttpURL(input.into()),
            other => todo!(),
        },
        // This is a file
        Err(ParseError::RelativeUrlWithoutBase) => LocatorWithProtocol::LocalFilePath(input.into()),
        Err(e) => todo!(), //e,
    }
}

enum LocatorWithProtocol {
    LocalFilePath(protocol::LocalFilePath),
    HttpURL(protocol::HttpURL),
}

mod protocol {
    pub struct LocalFilePath(pub String);
    pub struct HttpURL(pub String);

    impl From<&str> for LocalFilePath {
        fn from(value: &str) -> Self {
            Self(value.to_owned())
        }
    }

    impl From<&str> for HttpURL {
        fn from(value: &str) -> Self {
            Self(value.to_owned())
        }
    }
}

#[cfg(not(target_family = "wasm"))]
#[async_trait]
impl GetReader for protocol::LocalFilePath {
    type R = tokio::fs::File;

    async fn get(&self) -> Self::R {
        tokio::fs::File::open(&self.0).await.unwrap()
    }
}

lazy_static! {
    static ref CLIENT: reqwest::Client = reqwest::Client::new();
}

#[cfg_attr(target_family = "wasm", async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait)]
impl GetReader for protocol::HttpURL {
    type R = crate::http::HTTPFile;

    async fn get(&self) -> Self::R {
        HTTPFile::new(CLIENT.clone(), self.0.clone()).await.unwrap()
    }
}
