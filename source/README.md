Make sure to take a look at [`ARCHITECTURE.md`](../ARCHITECTURE.md) for a detailed overview of how Carton works and where various features are implemented.

Carton is made up of the following crates:

- `carton`: The core Carton library implemented in  Rust
- `carton-bindings-*`: Bindings for other programming languages (e.g python, wasm, nodejs)
- `carton-runner-interface`: The runner interface is how Carton communicates with runners. See ARCHITECTURE.md for more details.
- `carton-runner-packager`: A library that can package, fetch, and discover runners. This also includes a binary used in CI to upload nightly releases of the runners.
- `carton-runner-*`: Implementations of individual runners (e.g. python, torchscript, etc)
- `carton-utils`: Utilities for dealing with downloads, config, caching, and archives.
- `carton-utils-py`: Shared functionality between the python bindings and the python runners
- `anywhere`: A library that can serve a Lunchbox filesystem over a transport. Used by the runner interface to expose models to runners.
- `carton-macros`: Procedural rust macros used by other crates
- `fetch-deps`: Used as part of the build to download dependencies like libtorch
