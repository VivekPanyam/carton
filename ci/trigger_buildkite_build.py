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

# This script triggers a buildkite build for the current SHA and branch, waits for it to complete, and then
# downloads artifacts. It looks at GITHUB_* env vars and is intended to be run from github actions.
#
# To use, make sure requests is installed (pip3 install requests)
import os
import requests
import time


BUILDKITE_TOKEN = os.getenv("BUILDKITE_TOKEN")
if BUILDKITE_TOKEN is None:
    raise ValueError("Expected BUILDKITE_TOKEN env var to be set")

GITHUB_SHA = os.getenv("GITHUB_SHA")
GITHUB_REF_TYPE = os.getenv("GITHUB_REF_TYPE")

if GITHUB_REF_TYPE != "branch":
    raise ValueError("Expected GITHUB_REF_TYPE to be 'branch'")

GITHUB_BRANCH = os.getenv("GITHUB_REF_NAME")

# Trigger a build

res = requests.post("https://api.buildkite.com/v2/organizations/carton/pipelines/nightly/builds", headers={"Authorization": f"Bearer {BUILDKITE_TOKEN}"}, json={
    "commit": GITHUB_SHA,
    "branch": GITHUB_BRANCH,
    "message": "Nightly build",
})

data = res.json()
build_url = data["url"]
build_number = data["number"]

print(f"Created build {data['web_url']}")

while True:
    print("Waiting for build to complete...")

    # Get the build status
    data = requests.get(build_url, headers={"Authorization": f"Bearer {BUILDKITE_TOKEN}"}).json()
    state = data["state"]

    if state == "passed":
        break

    # States that are okay, but we need to keep checking
    if state in ["creating", "scheduled", "running"]:
        # Sleep for a bit before rechecking
        time.sleep(10.0)
    else:
        raise ValueError(f"The build failed with status: {state}")

print("Build complete! Getting artifacts...")

fetch_url = f"https://api.buildkite.com/v2/organizations/carton/pipelines/nightly/builds/{build_number}/artifacts"
while fetch_url is not None:
    res = requests.get(fetch_url, headers={"Authorization": f"Bearer {BUILDKITE_TOKEN}"})

    # Get the next url if any
    if 'next' in res.links:
        fetch_url = res.links['next']['url']
    else:
        fetch_url = None

    artifacts = res.json()
    for artifact in artifacts:
        download_url = artifact["download_url"]
        dirname = artifact["dirname"]
        filename = artifact["filename"]
        target_dir = os.path.join("/tmp", dirname)
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, filename)

        print(f"Downloading {filename} to {target_path} ...")

        res = requests.get(download_url, headers={"Authorization": f"Bearer {BUILDKITE_TOKEN}"})
        with open(target_path, "wb") as f:
            f.write(res.content)
