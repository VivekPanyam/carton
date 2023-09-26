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

import os
import torch
import asyncio
import logging
import numpy as np
import cartonml as carton

from cartonml import TensorSpec, Example, SelfTest
from cartonml.utils.hf import get_linked_files
from diffusers import DiffusionPipeline

MODEL_DESCRIPTION = """
Stable Diffusion XL 1.0 base combined with the refiner as an ensemble of experts.

![](https://raw.githubusercontent.com/Stability-AI/generative-models/477d8b9a7730d9b2e92b326a770c0420d00308c9/assets/000.jpg)

Options that can be passed when loading the model:

 - `model.compile` (bool): whether or not to use `torch.compile`. Defaults to false
 - `model.enable_cpu_offload` (bool): whether or not to use [`enable_model_cpu_offload()`](https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_xl#diffusers.StableDiffusionXLPipeline.enable_model_cpu_offload). Defaults to false
"""

async def main():
    # Configure logging format
    FORMAT = '[%(asctime)s %(levelname)s %(name)s] %(filename)s:%(lineno)d %(message)s'
    logging.basicConfig(format=FORMAT)

    # We want trace messages to show up
    logging.getLogger().setLevel(5)

    # Download the models we need
    base_revision = "bf714989e22c57ddc1c453bf74dab4521acb81d8"
    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        revision = base_revision,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        cache_dir="./to_pack/model",
    )

    refiner_revision = "93b080bbdc8efbeb862e29e15316cff53f9bef86"
    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        revision = refiner_revision,
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        cache_dir="./to_pack/model",
    )

    # For a smaller output model file, we can let Carton know to pull large files from HF instead of storing
    # them in the output file. This can also lead to better caching behavior if the same files are used across
    # several cartons
    linked_files = get_linked_files("stabilityai/stable-diffusion-xl-base-1.0", base_revision)
    for sha256, url in get_linked_files("stabilityai/stable-diffusion-xl-refiner-1.0", refiner_revision).items():
        linked_files[sha256] = url

    # Pack the model with carton
    with open('./lion.png', 'rb') as f:
        lion_output = f.read()

    with open('./elephant.png', 'rb') as f:
        elephant_output = f.read()

    with open('./rapper.png', 'rb') as f:
        rapper_output = f.read()

    with open('./ship.png', 'rb') as f:
        ship_output = f.read()

    packed_model_path = await carton.pack(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "to_pack"),
        runner_name = "python",
        required_framework_version="=3.10",
        runner_opts = {
            "entrypoint_package": "infer",
            "entrypoint_fn": "get_model",
            "model.compile": False,
            "model.enable_cpu_offload": False,
        },
        model_name = "sdxl_1_with_refiner_eoe",
        short_description = "Stable Diffusion XL 1.0 + refiner as an Ensemble of Experts",
        model_description = MODEL_DESCRIPTION,
        license = "openrail++",
        homepage = "https://github.com/Stability-AI/generative-models",
        inputs = [
            TensorSpec(
                name = "prompt",
                dtype = "string",
                shape = [],
                description = "The prompt for the model."
            ),
            TensorSpec(
                name = "n_steps",
                dtype = "uint32",
                shape = [],
                description = "Optional. The number of inference steps. Defaults to 40"
            ),
            TensorSpec(
                name = "high_noise_frac",
                dtype = "float32",
                shape = [],
                description = "Optional. The proportion of `n_steps` to run the base model for. Must be a number between 0 and 1. Defaults to 0.8"
            ),
        ],
        outputs = [
            TensorSpec(
                name = "image",
                dtype = "uint8",
                shape = ["height", "width", "num_channels"],
                description = "The generated image as an HWC tensor"
            ),
        ],
        examples = [
            Example(
                inputs = dict(prompt = np.array("A majestic lion jumping from a big stone at night")),
                sample_out = dict(image = lion_output)
            ),
            Example(
                inputs = dict(prompt = np.array("A majestic elephant on a big stone at night")),
                sample_out = dict(image = elephant_output)
            ),
            Example(
                inputs = dict(prompt = np.array("A rapper on stage at a festival, 4k HDR")),
                sample_out = dict(image = rapper_output)
            ),
            Example(
                inputs = dict(prompt = np.array("miniature sailing ship sailing in a heavy storm inside of a horizontal glass globe inside on a window ledge golden hour, home photography, 50mm, Sony Alpha a7")),
                sample_out = dict(image = ship_output)
            )
        ],
        self_tests = [
            SelfTest(
                name = "quickstart",
                inputs = dict(prompt = np.array("A majestic lion jumping from a big stone at night")),
            )
        ],
        linked_files = linked_files,
    )

    print("Packed path", packed_model_path)

    model = await carton.load(
        packed_model_path,
        override_runner_opts = {
        "model.enable_cpu_offload": True
    })

    # out = await model.infer({
    #     "prompt": np.array("miniature sailing ship sailing in a heavy storm inside of a horizontal glass globe inside on a window ledge golden hour, home photography, 50mm, Sony Alpha a7")
    # })

    # global image
    # image = out["image"]

asyncio.run(main())