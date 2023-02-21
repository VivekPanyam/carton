import unittest
import cartonml as carton
import numpy as np
import urllib.request

from cartonml import TensorSpec, SelfTest, Example
import tempfile

MODEL_DESCRIPTION = """
Some description of the model that can be as detailed as you want it to be

This can span multiple lines. It can use markdown, but shouldn't use arbitrary HTML
(which will usually be treated as text instead of HTML when this is rendered)

You can use images and links, but the paths must either be complete https
URLs (e.g. https://example.com/image.png) or must reference a file in the misc folder (e.g `@misc/file.png`).
See the `misc_files` section of this document for more details

![Model Architecture](@misc/model_architecture.png)
"""

class Test(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Grab an image from the Penn-Fudan pedestrian detection database
        with urllib.request.urlopen('https://www.cis.upenn.edu/~jshi/ped_html/images/PennPed00015_1.png') as f:
            self.input_image = f.read()

        # Grab the whisper approach graphic
        with urllib.request.urlopen('https://raw.githubusercontent.com/openai/whisper/main/approach.png') as f:
            self.model_architecture = f.read()

    def test_roundtrip_tensorspec(self):
        """
        Ensure that python -> rust -> python doesn't change anything for TensorSpecs
        """
        items = [
            dict(name = "a", dtype = "float32", shape = None),
            dict(name = "b", dtype = "float64", shape = "some_symbol"),
            dict(name = "c", dtype = "string", shape = [None]),
            dict(name = "d", dtype = "int8", shape = ["some_symbol"]),
            dict(name = "e", dtype = "int16", shape = []),
            dict(name = "f", dtype = "int32", shape = ["batch_size", 3, 512, 512]),
            dict(name = "g", dtype = "int64", shape = [None, None, None]),
            dict(name = "h", dtype = "uint8", shape = [2, 3, 512, 512]),
            dict(name = "i", dtype = "uint16", shape = ["multiple", "symbols"]),
            dict(name = "j", dtype = "uint32", shape = [2, None, "symbol"]),
            dict(name = "long", dtype = "uint64", shape = None, description = "A description"),
        ]
        
        for item in items:
            ts = TensorSpec(**item)

            self.assertEqual(ts.name, item["name"])
            self.assertEqual(ts.dtype, item["dtype"])
            self.assertEqual(ts.shape, item["shape"])
            if "description" in item:
                self.assertEqual(ts.description, item["description"])

    async def test_roundtrip_selftest(self):
        """
        Ensure that python -> rust -> python doesn't change anything for SelfTests
        """
        items = [
            dict(
                name = "a_self_test",
                description = "A self test",
                inputs = dict(a = np.ones(5, dtype=np.float32)),
                expected_out = dict(y = np.ones(5, dtype=np.uint32))
            ),
            dict(
                name = "2 ",
                inputs = dict(a = np.random.rand(5)),
                expected_out = dict(y = np.random.randint(1, 5, 5, dtype=np.uint32))
            )
        ]

        for item in items:
            st = SelfTest(**item)
            self.assertEqual(st.name, item["name"])

            for k, v in item["inputs"].items():
                np.testing.assert_equal(await st.inputs[k].get(), v)

            for k, v in item["expected_out"].items():
                np.testing.assert_equal(await st.expected_out[k].get(), v)

            if "description" in item:
                self.assertEqual(st.description, item["description"])

    async def test_roundtrip_example(self):
        """
        Ensure that python -> rust -> python doesn't change anything for Examples
        """
        items = [
            dict(
                name = "an_optional_name",
                description = "An example of an image input",
                inputs = dict(f = self.input_image),
                sample_out = dict(y = np.array(5))
            ),
            dict(
                name = " s o me na me wi th spaces",
                inputs = dict(f = np.array(5)),
                sample_out = dict(y = self.model_architecture, z = np.random.rand(5))
            ),
        ]

        for item in items:
            ex = Example(**item)
            self.assertEqual(ex.name, item["name"])

            for k, v in item["inputs"].items():
                if isinstance(v, np.ndarray):
                    np.testing.assert_equal(await ex.inputs[k].get(), v)
                else:
                    self.assertEqual(v, await ex.inputs[k].read())

            for k, v in item["sample_out"].items():
                if isinstance(v, np.ndarray):
                    np.testing.assert_equal(await ex.sample_out[k].get(), v)
                else:
                    self.assertEqual(v, await ex.sample_out[k].read())

            if "description" in item:
                self.assertEqual(ex.description, item["description"])

    async def test_load_unpacked(self):
        """
        Test of load_unpacked that uses most (if not all) of the documented arguments
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

        model = await carton.load_unpacked(
            dir,
            runner_name = "python",
            required_framework_version = "=3.11",
            runner_opts = {
                "entrypoint_package": "main",
                "entrypoint_fn": "get_model",
            },
            model_name = "Test Model",
            short_description = "A short description that should be less than or equal to 100 characters.",
            model_description = MODEL_DESCRIPTION,
            required_platforms = [
                "x86_64-unknown-linux-gnu",
                "aarch64-unknown-linux-gnu",
                "x86_64-apple-darwin",
                "aarch64-apple-darwin",
            ],
            inputs = [
                TensorSpec(
                    name = "a",
                    dtype = "float32",
                    shape = None,
                ),
                TensorSpec(
                    name = "b",
                    dtype = "float64",
                    shape = "some_symbol",
                ),
                TensorSpec(
                    name = "c",
                    dtype = "string",
                    shape = [None],
                ),
                TensorSpec(
                    name = "d",
                    dtype = "int8",
                    shape = ["some_symbol"],
                ),
                TensorSpec(
                    name = "e",
                    dtype = "int16",
                    shape = [],
                ),
                TensorSpec(
                    name = "f",
                    dtype = "int32",
                    shape = ["batch_size", 3, 512, 512],
                ),
                TensorSpec(
                    name = "g",
                    dtype = "int64",
                    shape = [None, None, None],
                ),
            ],
            outputs = [
                TensorSpec(
                    name = "y",
                    dtype = "uint32",
                    shape = [2, 3, 512, 512],
                    description = "Some description of the output"
                )
            ],
            self_tests = [
                SelfTest(
                    name = "a_self_test",
                    description = "A self test",
                    inputs = dict(a = np.ones(5, dtype=np.float32)),
                    expected_out = dict(y = np.ones(5, dtype=np.uint32))
                )
            ],
            examples = [
                Example(
                    name = "an_optional_name",
                    description = "An example of an image input",
                    inputs = dict(f = self.input_image),
                    sample_out = dict(y = np.array(5))
                )
            ],
            misc_files = {
                "model_architecture.png": self.model_architecture
            },
            visible_device = "CPU"
        )

        out = await model.infer({})

        major, minor = out["out"]
        if major != 3 or minor != 11:
            raise ValueError(f"Got an unexpected version of python. Got {major}.{minor} and expected 3.11")
        
        # Extract the example image from the model
        example_image = await model.info.examples[0].inputs["f"].read()
        self.assertEqual(self.input_image, example_image)

        self_test_out = await model.info.self_tests[0].expected_out["y"].get()
        np.testing.assert_equal(np.ones(5, dtype=np.uint32), self_test_out)

        # Extract the model architecture image
        model_arch_readback = await model.info.misc_files["model_architecture.png"].read()
        self.assertEqual(self.model_architecture, model_arch_readback)

if __name__ == "__main__":
    unittest.main()