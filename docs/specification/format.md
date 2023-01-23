This document describes v1 of the carton file format specification.

# Carton format

Every carton is a zipfile with the extension `.carton`. The supported zip compression methods are `Stored`, `Deflate`, and `zstd`.

Each carton contains the following:

- A file named `carton.toml`
- A file named `MANIFEST`
- A folder named `model`
- An optional folder named `tensor_data`
- An optional folder named `misc`
- An optional file named `LINKS`

These are described in more detail below.

## `carton.toml`

This is a toml file with the following structure

```toml
# A number defining the carton spec version. This is currently 1
spec_version = 1

# Optional, but highly recommended
# The name of the model
model_name = "super_awesome_test_model"

# Optional
model_description = """
Some description of the model that can be as detailed as you want it to be

This can span multiple lines. It can use markdown, but shouldn't use arbitrary HTML
(which will usually be treated as text instead of HTML when this is rendered)

You can use images and links, but the paths must either be complete https
URLs (e.g. https://example.com/image.png) or must reference a file in the misc folder (e.g `@misc/file.png`).
See the misc section of this document for more details
"""

# Optional
# A list of platforms the model supports. If empty or unspecified, all platforms are okay.
# The contents of this list are target triples. For example:
# x86_64-unknown-linux-gnu
# aarch64-unknown-linux-gnu
# x86_64-apple-darwin
# aarch64-apple-darwin
required_platforms = []

# Specifying inputs and outputs is optional, but highly recommended to make
# it easy to use your model.
# If you specify one input or output, you must specify all inputs and outputs
# Note that the input and output structures have an ordering (they're arrays of toml tables)
# https://toml.io/en/v1.0.0#array-of-tables
# This ordering is used if the code running the model does not specify tensor names when running
# inference.
[[input]]
# A string
name = "x"

# The dtype. Valid values are
# - float32
# - float64
# - string
# - int8
# - int16
# - int32
# - int64
# - uint8
# - uint16
# - uint32
# - uint64
dtype = "float32"

# Shape is either:
# - a symbol (an arbitrary string that must resolve to the same value wherever it's
#   used across all inputs and outputs). "*" is a reserved symbol name. It can have
#   a different value anytime it's used. See below for examples.
# - The string "*", meaning any shape is allowed
# - An empty list, meaning the value is a scalar
# - A list with one entry per dimension. Each entry can be a symbol or an integer
shape = ["batch_size", 3, 512, 512]

# For example, a tensor with 3 dims of any value
# Note: this also supports ragged tensors / nested tensors
# shape = ["*", "*", "*"]

# Other examples
# Any number of dims are allowed:
#   shape = "*"
# A scalar:
#   shape = []
# A symbol for the overall shape:
#   shape = "input_shape"

# Optional description of the input
description = "Something"

# Optional
# All inputs and outputs can have an "internal name" as well
# For example, this might be used to remap a user-visible name to a node
# in a TF or PyTorch graph. Carton doesn't expose these names to callers, but
# runners can use them
internal_name = "some_namespace/in_x:0"

# Another input
[[input]]
name = "y"
dtype = "float32"
shape = ["batch_size", 10]

# Same structure as inputs
[[output]]
name = "out"
shape = ["batch_size", 128]
dtype = "float32"

# Optional
# You can provide test data as well to enable basic self-tests
# These reference tensors stored in the `tensor_data` folder
# Usually this section is not handwritten
# `inputs` and `outputs` must be specified above in order to specify test data
[[self_test]]
# Optional
name = ""

# Optional
description = ""

# Required if this section is specified
inputs = { x = "@tensor_data/self_test_tensor_0", y = "@tensor_data/sometensor"}

# Optional
# If this is specified, the output will be compared using
# something conceptually similar to np.allclose
expected_out = { out = "@tensor_data/another_tensor"}

# Optional
# Finally, you can also provide example inputs and outputs
# `inputs` and `outputs` must be specified above in order to specify this section
# These examples don't need to be runnable and can reference images, audio, etc (in
# `misc` folder)
[[example]]
# Optional
name = ""

# Optional
description = ""

# These can reference things in the `tensor_data` or `misc`
inputs = { x = "@misc/input.png", y = "@tensor_data/something" }

sample_out = { out = "@misc/..." }

[runner]
# A string with the name of the runner required to run this model
runner_name = "torchscript"

# A version specifier of the allowed framework versions. In most cases, this should be exactly one version
# These should be semver version requirements parsable by https://docs.rs/semver/1.0.16/semver/struct.VersionReq.html
required_framework_version = "=1.12.1"

# This is basically a version for the contents of the `model` folder
# Generally not user visible
runner_compat_version = 2

# Optional
# Nothing here must be required to run the model
# Runners are required to be able to correctly run the model even if this section is removed
# These options are handled by the runner (the "torchscript" runner in this case) and are different
# for every runner
# These can be overridden at runtime.
# Runners should warn if they are passed options they don't understand (not just silently ignore them)
[runner.opts]
# For example, if we want to set threading configuration for running this model, we may set
# https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html#runtime-api
num_interop_threads = 4
num_threads = 1
```

Any unknown tables or fields are ignored. This lets us add additional data in the future without having to bump the `spec_version`

## `MANIFEST`

This is a file that lists the contents of the carton in alphabetical order with one entry per line. Each line is in the following format:

```
{filepath}={sha256}
```
No spaces or other characters are allowed. The only two files it does not contain an entry for are `MANIFEST` (i.e. itself) and `LINKS` (if any).


For example:

```
carton.toml=712f4a7aa03b1ba5ac4e6c3ee90ec2ece4cc2f78470d7ef40dda9e96e356ad4a
model/model.xgboost=e550f6224a5133f597d823ab4590f369e0b20e3c6446488225fc6f7a372b9fe2
```

Note: Any changes to the manifest format require a bump in `spec_version` above

The purpose of this file is so we have an easy way to get a unique hash for any carton without having to fetch or process the whole thing. It also helps build efficient caching.

*Note: "model hash" or "manifest hash" generally refers to the sha256 of the MANIFEST file.*

## `model`

The model folder contains the model and/or whatever other information the runner needs to load the model. The contents of this folder are unspecified and vary across runners.

## `tensor_data`

The tensor_data folder is optional and contains test data and/or example data referenced by the `carton.toml` file.

This folder contains an `index.toml` file of the following format. This file must exist if there are any other files in this directory.

```toml
[[tensor]]
name = "some_string_tensor"
dtype = "string"
shape = [1, 2, 3]
file = "tensor_0.toml"

[[tensor]]
name = "some_numeric_tensor"
dtype = "float32"
shape = [2, 2, 3]
file = "tensor_1.bin"

[[tensor]]
name = "some_nested_tensor"

# Nested/ragged tensors
dtype = "nested"

# Note: The inner tensors must not be nested tensors.
inner = [
    "some_other_tensor",
    "another_tensor",
]

# ...
```

Numeric tensors are stored in `.bin` files as little-endian, contiguous, C-order tensors.

String tensors are stored in `toml` files (one for each string tensor)

```toml
data = ["a", "b", "c", "d", "e"]
```

Rationale:
1. This lets us avoid a binary format that we have to carefully manage backwards compatibility for (e.g. a Rust struct serialized with bincode)
2. The main overhead is the repeated `", "` for large tensors. In cases where it matters, this can be mitigated by zipfile compression. Most likely, this overhead won't have an impact. There's a relatively minor difference between storing strings in a binary format or storing them in a text format (vs the major difference between "3.14159265359" and the same value as a `float32` or `float64`).

Because the data for each tensor is stored in a separate file, this structure enables fine-grained lazy loading of tensor data. Other options like safetensors were considered, but using them proved challenging. Safetensors store all the tensors in a single file that can be mmapped and lazy loaded. Unfortunately, the zipfile structure of a carton removes this possibility and requires loading the entire safetensors file.

## `misc`

Non-tensor data referenced by the examples. No specific requirements on file types, but they generally should be commonly used types. Systems using this data should generally understand at least the following types:
 - png
 - jpeg
 - mp3
 - mp4

## `LINKS`

Storing a large number of models in a repository can be inefficient if there are many duplicated files. To help with this, we define a `LINKS` file. This is a toml file mapping sha256s to a list of URLs where that file can be fetched.

```toml
version = 1

[urls]
e550f6224a5133f597d823ab4590f369e0b20e3c6446488225fc6f7a372b9fe2 = ["https://.../file"]

```

This enables the zipfile to not include all the files in the manifest. If there are files in `MANIFEST` that are missing from the zip file, `LINKS` must contain them.

The `LINKS` file is only visible to and used by the loader; most of the carton library will see the structure after links have been resolved.

As such, the `LINKS` file should **NOT** be included in the manifest.

*Note: This means that the existence or contents of `LINKS` does not affect the model hash of the carton.*

Also note that the files listed in `LINKS` may be compressed (identified by the `content-encoding` header of the response):

- Readers should support `zstd`, `gzip`, `br` (brotli), and `deflate` (deflate) (Most HTTP libraries will transparently handle this, but you may have to explicitly handle `zstd`)

The sha256 is of the original file - not the compressed one.

---

Because of this format, the only required user-specified fields are `runner_name` and `required_framework_version`. This has the nice benefit of letting us load unpacked models just given those two pieces of info.