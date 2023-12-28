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

import json
import os
import tempfile
from typing import List

import cartonml as carton
from cartonml import TensorSpec

import jax
import jax.numpy as jnp

async def pack_jax_for_xla(model, inputs: List[TensorSpec], outputs: List[TensorSpec], **kwargs) -> str:
    """
    Pack a JAX model supported by `jax.jit` into a Carton.
    """
    # Create a temp dir
    with tempfile.TemporaryDirectory() as temp_dir:
        # Wrap the JAX model in a function with args in the correct order
        # Based on https://github.com/google/jax/blob/main/jax/tools/jax_to_ir.py (Apache 2.0)
        args = [jnp.zeros(s.shape, s.dtype) for _, s in inputs]

        # Wrapper that takes in args in the order of `input_shapes` and converts them
        # to kwargs for calling `fn`.
        # It then transforms the outputs to return in order as well
        def ordered_wrapper(*args):
            arg_names = [item.name for item in inputs]
            out_dict = model(**dict(zip(arg_names, args)))

            return [out_dict[item.name] for item in outputs]

        # Export to HLO
        comp = jax.xla_computation(ordered_wrapper)(*args)
        serialized_proto = comp.as_serialized_hlo_module_proto()

        with open(os.path.join(temp_dir, "model.pb"), 'wb') as f:
            f.write(serialized_proto)

        # Write metadata that stores the arg ordering
        with open(os.path.join(temp_dir, "model.json"), 'w') as f:
            json.dump(dict(
                input_ordering=[item.name for item in inputs],
                output_ordering=[item.name for item in outputs],
            ), f)

        return await carton.pack(
            temp_dir,
            inputs = inputs,
            runner_name = "xla",
            # TODO: Allow users to set the version of XLA they want to run with
            # How is XLA versioned? I don't see anything standard version numbers with the prebuilts
            required_framework_version = ">= 0.0.0"
            **kwargs
        )
