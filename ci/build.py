# Copyright 2023 Vivek Panyam
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Build and test. Intented to be used in CI
import subprocess
import argparse
import os
import sys
import toml
from datetime import datetime

def run_command(cmd, **kwargs):
    # Filter out `None`
    cmd = [item for item in cmd if item is not None]
    print(f"Running: {cmd}", flush=True)
    if os.getenv("GITHUB_ACTIONS") is not None:
        print(f"::group::Running {' '.join(cmd)}", flush=True)

    kwargs["stdout"] = sys.stdout
    kwargs["stderr"] = subprocess.STDOUT
    if HIDE_OUTPUT:
        kwargs["stdout"] = subprocess.DEVNULL
        kwargs["stderr"] = subprocess.DEVNULL

    subprocess.check_call(cmd, **kwargs)

    if os.getenv("GITHUB_ACTIONS") is not None:
        print("::endgroup::", flush=True)

def update_version_numbers():
    """
    Updates version numbers for dev builds
    """
    # TODO: make this more robust
    bindings_config = "source/carton-bindings-py/Cargo.toml"
    data = toml.load(bindings_config)
    data["package"]["version"] += datetime.utcnow().strftime("-dev%Y%m%d")
    with open(bindings_config, 'w') as f:
        toml.dump(data, f)

    # Update the python package name
    package_config = "source/carton-bindings-py/pyproject.toml"
    data = toml.load(package_config)
    data["project"]["name"] = "cartonml-nightly"
    with open(package_config, 'w') as f:
        toml.dump(data, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = 'Carton CI Build Script')
    parser.add_argument("-t", "--target", required=True, help="The target triple to build against")
    parser.add_argument("--release", action="store_true", help="Build with `--release`")
    parser.add_argument("--nightly", action="store_true", help="Update version numbers with a dev build suffix")
    parser.add_argument("--hide_output", action="store_true", help="Redirect all subprocess output to /dev/null")
    parser.add_argument("--runner_release_dir", help="The runner release dir (if any)")
    args = parser.parse_args()

    TARGET = args.target
    RELEASE_FLAG = "--release" if args.release else None
    HIDE_OUTPUT = args.hide_output

    print(f"""
    Building with configuration:
        Target: `{TARGET}`
        Release mode: {args.release}
        Nightly mode: {args.nightly}
        Runner release dir: {args.runner_release_dir}
    """)

    # Check up front so we don't build and then fail
    if args.runner_release_dir is not None:
        if os.path.exists(args.runner_release_dir):
            raise ValueError("Supplied runner release dir already exists! Please remove it and try again.")

    # Update version numbers if this is a nightly build
    if args.nightly:
        update_version_numbers()

    # Warnings as errors in CI
    os.environ["RUSTFLAGS"] = "-Dwarnings"

    # This is hacky. TODO: improve this
    if TARGET == "aarch64-unknown-linux-gnu" and sys.platform in ["linux", "linux2"]:
        os.environ["LIBTORCH_CXX11_ABI"] = "0"

    # Fetch deps (always in release mode)
    run_command(["cargo", "run", "--timings", "--release", "-p", "fetch-deps", "--target", TARGET])

    # Build everything
    run_command(["cargo", "build", RELEASE_FLAG, "--verbose", "--timings", "--target", TARGET])

    # Build wheels for the python bindings
    # TODO: store timing info
    py_bindings_cmd = [sys.executable, "-m", "maturin", "build", RELEASE_FLAG, "--target", TARGET]
    if "linux" in TARGET:
        py_bindings_cmd += ["--compatibility", "manylinux_2_28"]

    run_command(py_bindings_cmd, cwd=os.path.join(os.getcwd(), "source/carton-bindings-py"))

    # Run tests
    run_command(["cargo", "test", RELEASE_FLAG, "--verbose", "--timings", "--target", TARGET], env=dict(os.environ, RUST_LOG="info,carton=trace"))

    # Build the runner releases
    if args.runner_release_dir is not None:
        os.makedirs(args.runner_release_dir)
        run_command(["cargo", "run", RELEASE_FLAG, "--timings", "--target", TARGET, "-p", "carton-runner-py", "--bin", "build_releases", "--", "--output-path", args.runner_release_dir])
        run_command(["cargo", "run", RELEASE_FLAG, "--timings", "--target", TARGET, "-p", "carton-runner-rust-bert", "--bin", "build_rust_bert_releases", "--", "--output-path", args.runner_release_dir])
        run_command(["cargo", "run", RELEASE_FLAG, "--timings", "--target", TARGET, "-p", "carton-runner-torch", "--bin", "build_torch_releases", "--", "--output-path", args.runner_release_dir])
        run_command(["cargo", "run", RELEASE_FLAG, "--timings", "--target", TARGET, "-p", "carton-runner-wasm", "--bin", "build_wasm_releases", "--", "--output-path", args.runner_release_dir])

    # Show sccache stats
    RUSTC_WRAPPER = os.getenv("RUSTC_WRAPPER", "")
    if "sccache" in RUSTC_WRAPPER:
        run_command([RUSTC_WRAPPER, "--show-stats"])
