from .cartonml import *

# Load a model with a type
# carton.load("model.xgboost", runner="xgboost")
# carton.load("model.savedmodel", runner="tf")
# carton.load("model.lightgbm", runner="lightgbm")
# carton.load("model.trt", runner="tensorrt")
# carton.load("model.pt", runner="torchscript")
# carton.load("/path/to/python/script.py", runner="python")
# def load(path: str, runner: str, runner_version: str, runner_opts: str, visible_device: str):
#     pass

# Allowed devices:
# CPU
# GPU0
# GPU1
# GPU2
# GPU2
# GPU3
# GPU4
# GPU5
# GPU6
# GPU7

# Note: Just because a device is visible to a runner, does not mean the runner or model will use it


# Models can support an optional `seal` method that takes in a dict and moves all tensors to the appropriate device

def set_intermediate_runner():
    """
    This causes all load data to be passed to a specified runner that is responsible for actually loading the model
    """
    pass

class Model:
    @property
    def name(self):
        pass

    @property
    def runner(self):
        """
        The runner that is running this model
        """
        pass

    @property
    def inputs(self):
        pass

    @property
    def outputs(self):
        pass

    async def seal(self, inputs): # Takes a dict
        """
        This can be called on a set of inputs before a call to `run` so data movement can happen in parallel with another
        infer call.

        Optional, but can make pipelines more efficient.
        """
        pass

    async def run(self, inputs):
        """
        Run the model on a set of inputs and get the outputs back. This is an async function
        """
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        pass

# Load a carton
# def load(carton_path: str):
#     pass

# Saving a carton basially just saves all the metadata passed to `load`
# The user specified values at load time overwrite the saved data