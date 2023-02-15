## Runner Interface

The runner interface is the interface between the core library and the runners.

A runner is built against one version of the runner interface. The core library supports *all* versions of the interface.

This allows runner binaries from old releases to run correctly with new versions of the core library.

This interface does not have a specification, but is implementation defined with careful rules around compatibility. See `source/carton-runner-interface/src/do_not_modify/README.md`.

Any breaking changes require a new version of the runner interface.

## Runner Discovery

Carton searches the carton runners directory (set by the `CARTON_RUNNER_DIR` environment variable; defaults to `/usr/local/carton_runners`) for `runner.toml` files.

These files describe at least one runner (usually exactly one):

```toml
# runner.toml
version = 1

[[runner]]
# The name of this runner (e.g. "torchscript", "python")
# If this is a third party runner, this MUST be prefixed with a unique namespace followed by a forward slash
# For example: "mythirdpartyname/tensorflow"
runner_name = ""

# The exact semver version of the framework that this runner supports (e.g. "1.13.1", "3.6.0")
# Must be parsable by https://docs.rs/semver/1.0.16/semver/struct.Version.html
framework_version = ""

# This isn't directly user visible, but it is used for compatibility checks.
# MUST be incremented on breaking changes. This is monotonic for a given `runner_name`
# (i.e. it does NOT reset when framework versions change)
runner_compat_version = 1

# The interface that should be used to communicate with this runner
runner_interface_version = 1

# The release date of the runner as an rfc3339 string
runner_release_date = "1979-05-27T07:32:00Z"

# A path to the runner binary. This can be absolute or relative to this file
runner_path = "../path/to/runner"

# A target triple. See below
platform = "x86_64-apple-darwin"
```

## Rules


1. A carton packed by a given runner `(runner_name, framework_version, runner_compat_version)` MUST be runnable by all runners with the same `(runner_name, framework_version, runner_compat_version)`.
    - *Note: this implies forward and backward compatibility*
2. The storage format expected by the runner is tied to the `runner_compat_version`, not the `framework_version`.
    - This implies that a carton with a given `(runner_name, runner_compat_version)` MUST be loadable by any runners with the same `(runner_name, runner_compat_version)`. The load may not succeed within the underlying framework (e.g. if the `framework_version` of the chosen runner doesn't actually support the model that is being loaded), but it must not fail because the runner does not understand the format of the stored model.
    - Any breaking changes to the model format must result in an increase in the `runner_compat_version`
    - The reason for this is to allow models to specify a `required_framework_version` range they support (e.g. a torchscript model that supports Torch `1.10.0` - `1.13.1`). The constraints above mean that any runner that meets the `runner_name` and `runner_compat_version` criteria within the specified `framework_version` range should be able to load the model.

Because the core carton library includes implementations of all versions of the runner_interface, any carton that was packaged by an official release can *always* be loaded in the future.

*Note: the toml structure above allows a runner to support multiple framework versions, frameworks, etc, but this functionality should probably not be used without careful consideration. Individually maintaining forward and backward compatibility could be tricky. Especially if there are other runners with the same framework, framework_version and runner_compat_version.*


## Runner Installation

The carton library fetches a list of official runners from a well known URL (TODO: specify) that looks like

```js
[
    {
        "runner_name": "",
        "id": "", // A hash combining each of the sha256s in download_info. This is an implementation detail that could change
        "framework_version": "",
        "runner_compat_version": 1,
        "runner_interface_version": 1,
        "runner_release_date": "1979-05-27T07:32:00Z",

        // A list of URLs to zip, tar, or tar.gz files
        // The files are downloaded and unpacked based on `relative_path` below
        // The resulting folder structure MUST have a runner.toml in the root directory
        // Being able to split downloads into multiple files allows us to keep releases small and makes it more
        // feasible to have a regular release schedule.
        "download_info": [
            {
                "url": "https://.../runner.zip",
                "sha256": "5595aae3252ff078d06c0d588e7d85b086ca9fbcebe8dda047f07bb35d4527b0",
                "relative_path": ""
            },
            {
                "url": "https://download.pytorch.org/libtorch/cu116/libtorch-shared-with-deps-1.13.1%2Bcu116.zip",
                "sha256": "5595aae3252ff078d06c0d588e7d85b086ca9fbcebe8dda047f07bb35d4527b0",
                "relative_path": "libtorch/"
            },
        ],

        // A platform string. For example:
        // x86_64-unknown-linux-gnu
        // aarch64-unknown-linux-gnu
        // x86_64-apple-darwin
        // aarch64-apple-darwin
        "platform": ""
    },
    // ...
]
```


The release date is used as a tiebreaker when carton decides to install a backend (preferring newer releases).

By default, runners are installed automatically as models are loaded. However, they may also be explicitly preinstalled:

```py
# TODO: implement
await carton.install_runner(framework="tensorflow", framework_version="2.1.0")
await carton.install_runner(framework="tensorflow", framework_version="2.1.0", runner_compat_version=1)
await carton.install_runner(framework="tensorflow", framework_version="2.1.0", sha256="5595aae3252ff078d06c0d588e7d85bcebe8dda0...")
await carton.install_runner(framework="tensorflow", sha256="5595aae3252ff078d06c0d588e7d85b086ca9fbcebe8dda0...")
await carton.install_runner(sha256="5595aae3252ff078d06c0d588e7d85b086ca9fbcebe8dda0...")
await carton.install_runner(url="https://example.com/some/runner.zip")

list_of_runners = await carton.get_installed_runners()

await carton.uninstall_runner(sha256="...")
```

If a sha256 is specified, that indiciates that we should use a particular release. An error will be thrown if the release doesn't work on the current platform or if the hash doesn't match the other args passed in.
