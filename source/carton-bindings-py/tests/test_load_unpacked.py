import asyncio
import carton
import tempfile

async def test():
    """
    Basic test of load_unpacked
    """
    dir = tempfile.mkdtemp()
    with open(f'{dir}/requirements.txt', 'w') as f:
        f.write("numpy")

    with open(f'{dir}/main.py', 'w') as f:
        f.write("""
import sys
import numpy as np

class Model:
    def __init__(self):
        pass

    def infer_with_tensors(self, tensors):
        return {
            "out": np.array([sys.version_info.major, sys.version_info.minor])
        }

def get_model():
    return Model()
        """)

    model = await carton.load_unpacked(dir, runner_name = "python", required_framework_version = "=3.11", opts = {
        "entrypoint_package": "main",
        "entrypoint_fn": "get_model",
    }, visible_device = "CPU")

    out = await model.infer_with_inputs({})

    major, minor = out["out"]
    if major != 3 or minor != 11:
        raise ValueError(f"Got an unexpected version of python. Got {major}.{minor} and expected 3.11")


asyncio.run(test())
