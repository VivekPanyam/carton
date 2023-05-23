# Utilities to pack huggingface transformers models
import tempfile
import transformers
import os
import sys
import dill
import numpy as np

from .. import pack

from typing import Callable, Any, Dict

from transformers import pipeline, Pipeline

from huggingface_hub import snapshot_download

async def pack_transformers_pipeline(pipeline_task: str, model_name: str, postproc: Callable[[Any], Dict[str, np.ndarray]]):
    """
    Packs a carton given the following:
     - The name of a transformers pipeline task (e.g. 'fill-mask')
     - A model to use for the pipeline (e.g. 'bert-base-uncased')
     - A callable that transforms the output of the pipeline into a dict mapping strings to numpy arrays (the output format of the overall model)

    Note: the callable must be pickle-able with `dill`
    """
    print("Fetching model from Hugging Face...")
    dir = tempfile.mkdtemp()
    model_path = os.path.join(dir, "data")

    # Fetch the model but ignore non-pytorch models
    snapshot_download(repo_id=model_name, local_dir=model_path, ignore_patterns=["flax_model.msgpack", "model.safetensors", "rust_model.ot", "tf_model.h5"])

    with open(f'{dir}/requirements.txt', 'w') as f:
        f.write(f"transformers=={transformers.__version__}\n")
        f.write(f"dill\n")
        f.write(f"torch\n")

    with open(f'{dir}/postproc.pkl', 'wb') as f:
        dill.dump(postproc, f, recurse=True, byref=True)

    with open(f'{dir}/main.py', 'w') as f:
            f.write(f"""
from transformers import pipeline

import torch
import dill

class Model:
    def __init__(self):
        # Carton will only make a cuda device visible if it's okay to use it
        # (i.e. a GPU was passed in as a `visible_device` when loading the model)
        device = 0 if torch.cuda.is_available() else -1

        with open('postproc.pkl', 'rb') as f:
            self.postproc = dill.load(f)

        self.pipeline = pipeline('{pipeline_task}', model="./data", tokenizer="./data", device=device)

    def infer_with_tensors(self, tensors):
        seq = list(tensors["input_sequences"])
        out = self.pipeline(seq)
        out = self.postproc(out)
        return out

def get_model():
    return Model()
        """)

    model_path = await pack(
        dir,
        runner_name="python",
        required_framework_version=f"={sys.version_info.major}.{sys.version_info.minor}",
        runner_opts = {
            "entrypoint_package": "main",
            "entrypoint_fn": "get_model",
        },
    )

    return model_path