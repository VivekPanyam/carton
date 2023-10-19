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

# This script is used to generate an XLA model used by the packing test
import os
import json
import jax
import jax.numpy as jnp

def model(a):
    """
    A "model" that doubles the input and returns it.
    Note that actual jax models for use with Carton are expected to return a dict.
    This is just for testing since we're manually defining the metadata for output names
    """
    return (a * 2, a * 3)


args = [jnp.zeros([5], jnp.float32)]

# Export to HLO
comp = jax.xla_computation(model)(*args)
serialized_proto = comp.as_serialized_hlo_module_proto()

with open("test_model.pb", 'wb') as f:
    f.write(serialized_proto)

# Write metadata that stores the arg ordering
with open("test_model.json", 'w') as f:
    json.dump(dict(
        input_ordering=["a"],
        output_ordering=["doubled", "tripled"],
    ), f)
