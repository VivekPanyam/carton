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