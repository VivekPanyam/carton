import os
import asyncio
import logging
import numpy as np
import cartonml as carton

from cartonml import TensorSpec, Example, SelfTest
from cartonml.utils.hf import get_linked_files
from transformers import DPTForDepthEstimation, DPTFeatureExtractor
from PIL import Image

MODEL_DESCRIPTION = """
DPT-Hybrid is model trained on 1.4 million images for monocular depth estimation. It was introduced in the paper [Vision Transformers for Dense Prediction](https://arxiv.org/abs/2103.13413) by Ranftl et al. (2021).  DPT uses the Vision Transformer (ViT) as backbone and adds a neck + head on top for monocular depth estimation.
"""

async def main():
    # Configure logging format
    FORMAT = '[%(asctime)s %(levelname)s %(name)s] %(filename)s:%(lineno)d %(message)s'
    logging.basicConfig(format=FORMAT)

    # We want trace messages to show up
    logging.getLogger().setLevel(5)

    # Download the models we need
    model_id = "Intel/dpt-hybrid-midas"
    model_revision = "fc1dad95a6337f3979a108e336932338130255a0"
    model = DPTForDepthEstimation.from_pretrained(
        model_id,
        revision = model_revision,
        cache_dir="./to_pack/model",
    )

    feature_extractor = DPTFeatureExtractor.from_pretrained(
        model_id,
        revision = model_revision,
        cache_dir="./to_pack/model",
    )

    # For a smaller output model file, we can let Carton know to pull large files from HF instead of storing
    # them in the output file. This can also lead to better caching behavior if the same files are used across
    # several cartons
    linked_files = get_linked_files(model_id, model_revision)

    # Pack the model with carton
    with open('./tim-rebkavets-B5PNmw5XSpk-unsplash.jpg', 'rb') as f:
        castle_image = f.read()

    with open('./castle_depth.png', 'rb') as f:
        castle_depth = f.read()

    # Create a small version of the input image for us to use in the quickstart self test
    example_input_resized = Image.open('./tim-rebkavets-B5PNmw5XSpk-unsplash.jpg')
    example_input_resized.thumbnail((512, 512), Image.Resampling.LANCZOS)
    example_input_as_array = np.asarray(example_input_resized)[np.newaxis, :]

    packed_model_path = await carton.pack(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "to_pack"),
        runner_name = "python",
        required_framework_version="=3.10",
        runner_opts = {
            "entrypoint_package": "infer",
            "entrypoint_fn": "get_model",
        },
        model_name = "intel_dpt_hybrid_midas",
        short_description = "DPT hybrid is a monocular depth estimation model",
        model_description = MODEL_DESCRIPTION,
        license = "Apache-2.0",
        homepage = "https://huggingface.co/Intel/dpt-hybrid-midas",
        inputs = [
            TensorSpec(
                name = "image",
                dtype = "uint8",
                shape = ["batch_size", "height", "width", "num_channels"],
                description = "The input image as an NHWC tensor"
            ),
        ],
        outputs = [
            TensorSpec(
                name = "depth",
                dtype = "uint8",
                shape = ["batch_size", "height", "width"],
                description = "The estimated depth map as an NHW tensor"
            ),
        ],
        examples = [
            Example(
                inputs = dict(image = castle_image),
                sample_out = dict(depth = castle_depth)
            ),
        ],
        self_tests = [
            SelfTest(
                name = "quickstart",
                inputs = dict(image = example_input_as_array),
            ),
        ],
        linked_files = linked_files,
    )

    print("Packed path", packed_model_path)

    # Testing
    model = await carton.load(packed_model_path)

    # out = await model.infer({
    #     "image": image
    # })

    # global depth_map
    # depth_map = out["depth"]

asyncio.run(main())