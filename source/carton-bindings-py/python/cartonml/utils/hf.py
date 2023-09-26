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

from collections import defaultdict
from huggingface_hub import HfApi, hf_hub_url
from typing import Dict, Optional

def get_linked_files(repo_id: str, revision: Optional[str]) -> Dict[str, list[str]]:
    """
    Get a mapping of all the LFS files in a huggingface repo. Using this as part of packing can help create
    smaller packed models and allow for more efficient caching.

    This mapping can be passed into `carton.pack`
    """
    _api = HfApi()
    repo_info = _api.repo_info(repo_id=repo_id, revision=revision, files_metadata=True)
    commit_hash = repo_info.sha

    lfs_mapping = defaultdict(list)
    for item in repo_info.siblings:
        file_path = item.rfilename
        if item.lfs is not None:
            lfs_mapping[item.lfs['sha256']].append(hf_hub_url(repo_id=repo_id, filename=file_path, revision=commit_hash))

    return lfs_mapping