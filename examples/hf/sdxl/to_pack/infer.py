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

from diffusers import DiffusionPipeline
import torch
import numpy as np

class Model:
    def __init__(self, compile = False, enable_cpu_offload = False):
        # Load the models we downloaded (notice the `local_files_only=True`)
        base_revision = "bf714989e22c57ddc1c453bf74dab4521acb81d8"
        base = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            revision = base_revision,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            cache_dir="./model",
            local_files_only=True
        )

        if not enable_cpu_offload and torch.cuda.is_available():
            base.to("cuda")

        if compile:
            base.unet = torch.compile(base.unet, mode="reduce-overhead", fullgraph=True)

        if enable_cpu_offload and torch.cuda.is_available():
            base.enable_model_cpu_offload()


        refiner_revision = "93b080bbdc8efbeb862e29e15316cff53f9bef86"
        refiner = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            revision = refiner_revision,
            text_encoder_2=base.text_encoder_2,
            vae=base.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
            cache_dir="./model",
            local_files_only=True
        )

        if not enable_cpu_offload and torch.cuda.is_available():
            refiner.to("cuda")

        if compile:
            refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)

        if enable_cpu_offload and torch.cuda.is_available():
            refiner.enable_model_cpu_offload()

        self.base = base
        self.refiner = refiner
        self.enable_cpu_offload = enable_cpu_offload

    def infer_with_tensors(self, tensors):
        n_steps = int(tensors.get("n_steps", 40))
        high_noise_frac = float(tensors.get("high_noise_frac", 0.8))

        prompt = tensors["prompt"].item()

        image = self.base(
            prompt=prompt,
            num_inference_steps=n_steps,
            denoising_end=high_noise_frac,
            output_type="latent",
        ).images

        if self.enable_cpu_offload and torch.cuda.is_available():
            self.base.to("cpu")
            torch.cuda.empty_cache()

        image = self.refiner(
            prompt=prompt,
            num_inference_steps=n_steps,
            denoising_start=high_noise_frac,
            image=image,
        ).images[0]

        if self.enable_cpu_offload and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "image": np.asarray(image)
        }

def get_model(compile: bool, enable_cpu_offload: bool, **kwargs):
    return Model(compile=compile, enable_cpu_offload=enable_cpu_offload)