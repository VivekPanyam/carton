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

# This script is used to generate a torchscript model used by the packing test
import torch
from typing import Dict, Any, List

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: Dict[str, Any]):
        a = inputs["a"]

        # Needed to help torch understand types
        assert isinstance(a, torch.Tensor)
        assert isinstance(inputs["b"], str)
        assert isinstance(inputs["c"], List[str])

        return {
            "doubled": a * 2,
            "string": "A string",
            "stringlist": ["A", "list", "of", "strings"]
        }

m = torch.jit.script(Model())
torch.jit.save(m, "test_model.pt")