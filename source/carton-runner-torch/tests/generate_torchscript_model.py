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