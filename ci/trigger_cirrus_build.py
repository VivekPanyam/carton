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

# This script triggers a cirrus CI build for the current SHA and branch, waits for it to complete, and then
# downloads artifacts. It looks at GITHUB_* env vars and is intended to be run from github actions.
#
# To use, make sure gql is installed (pip3 install gql[requests])
import os
import requests
import time
import zipfile
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport

CIRRUS_AUTH = os.getenv("CIRRUS_AUTH")
if CIRRUS_AUTH is None:
    raise ValueError("Expected CIRRUS_AUTH env var to be set")

transport = RequestsHTTPTransport("https://api.cirrus-ci.com/graphql", headers={"Authorization": CIRRUS_AUTH})

# Create a GraphQL client using the defined transport
client = Client(transport=transport, fetch_schema_from_transport=True)

# 1. Get the repository id
query = gql(
    """
    query {
        ownerRepository(platform: "github", owner: "VivekPanyam", name: "carton") {
            id
        }
    }
    """
)

# Execute the query on the transport and store the repo id
result = client.execute(query)
repository_id = result["ownerRepository"]["id"]

GITHUB_SHA = os.getenv("GITHUB_SHA")
GITHUB_REF_TYPE = os.getenv("GITHUB_REF_TYPE")

if GITHUB_REF_TYPE != "branch":
    raise ValueError("Expected GITHUB_REF_TYPE to be 'branch'")

GITHUB_BRANCH = os.getenv("GITHUB_REF_NAME")

# 1. Trigger a build for the provided sha
query = gql(
    """
    mutation CreateBuildDialogMutation($input: RepositoryCreateBuildInput!) {
        createBuild(input: $input) {
            build {
                id
            }
        }
    }
    """
)

CIRRUS_BUILD_CONFIG = """
env:
  CARGO_TERM_COLOR: always
  CI_NODE_INDEX: 0
  CI_NODE_TOTAL: 1

# Nightly release builds
# nightly_linux_task:
#   name: Nightly Release Build aarch64-unknown-linux-gnu
#   alias: nightly_linux
#   arm_container:
#     dockerfile: ci/ci.dockerfile
#     cpu: 4
#     memory: 16G
#   target_cache:
#     folder: target
#     fingerprint_script:
#       - rustc --version
#       - echo "nightly"
#       - cat Cargo.lock
#   build_and_test_script:
#     - pip3 install toml maturin==0.14.13
#     - python3 ci/build.py --target aarch64-unknown-linux-gnu --release --nightly --runner_release_dir $CIRRUS_WORKING_DIR/runner_releases
#   binaries_artifacts:
#     path: "runner_releases/*"
#   wheels_artifacts:
#     path: "target/wheels/*"
#   before_cache_script: rm -rf $CARGO_HOME/registry/index

nightly_macos_task:
  name: Nightly Release Build aarch64-apple-darwin
  alias: nightly_macos
  macos_instance:
    image: ghcr.io/cirruslabs/macos-ventura-xcode:14.2
  symlink_xcode_script:
    - ln -s /Applications/Xcode-14.2.0.app /Applications/Xcode.app
    - ls -alh /Applications/
  setup_script:
    - curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
  build_and_test_script:
    - source $HOME/.cargo/env
    - npm install -g yarn@1.22
    - pip3 install toml maturin==0.14.13
    - python3 ci/build.py --target aarch64-apple-darwin --release --nightly --runner_release_dir $CIRRUS_WORKING_DIR/runner_releases  --c_cpp_bindings_release_dir $CIRRUS_WORKING_DIR/bindings_releases
  binaries_artifacts:
    path: "runner_releases/*"
  wheels_artifacts:
    path: "target/wheels/*"
  bindings_artifacts:
    path: "bindings_releases/*"

"""

# Execute the query on the transport
result = client.execute(query, variable_values={"input": {
    "repositoryId": repository_id,
    "branch": GITHUB_BRANCH,
    "sha": GITHUB_SHA,
    "clientMutationId": "carton",
    "configOverride": CIRRUS_BUILD_CONFIG,
}})
print(result)
build_id = result["createBuild"]["build"]["id"]

while True:
    print("Waiting for build to complete...")

    # Get the build status
    query = gql(
        """
        query getBuild ($id: ID!) {
            build(id: $id) {
                status
            }
        }
        """
    )

    # Execute the query on the transport
    result = client.execute(query, variable_values={"id": build_id})
    print(result)
    if result["build"]["status"] == "COMPLETED":
        break

    if result["build"]["status"] in ["FAILED", "ABORTED", "ERRORED"]:
        status = result["build"]["status"]
        raise ValueError(f"The Cirrus CI build failed with status: {status}")

    time.sleep(10.0)

print("Build complete! Downloading runners...")
res = requests.get(f"https://api.cirrus-ci.com/v1/artifact/build/{build_id}/binaries.zip", headers={"Authorization": CIRRUS_AUTH})

with open("/tmp/binaries.zip", "wb") as f:
    f.write(res.content)

with zipfile.ZipFile("/tmp/binaries.zip", 'r') as zip_ref:
    zip_ref.extractall("/tmp")

print("Downloading wheels...")
res = requests.get(f"https://api.cirrus-ci.com/v1/artifact/build/{build_id}/wheels.zip", headers={"Authorization": CIRRUS_AUTH})

with open("/tmp/wheels.zip", "wb") as f:
    f.write(res.content)

with zipfile.ZipFile("/tmp/wheels.zip", 'r') as zip_ref:
    zip_ref.extractall("/tmp")


print("Downloading bindings...")
res = requests.get(f"https://api.cirrus-ci.com/v1/artifact/build/{build_id}/bindings.zip", headers={"Authorization": CIRRUS_AUTH})

with open("/tmp/bindings.zip", "wb") as f:
    f.write(res.content)

with zipfile.ZipFile("/tmp/bindings.zip", 'r') as zip_ref:
    zip_ref.extractall("/tmp")
