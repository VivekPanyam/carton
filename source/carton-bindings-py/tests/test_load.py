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

import asyncio
import carton
import numpy as np

async def test():
    model = await carton.load("/tmp/somepath", runner = "noop", runner_version = None, runner_opts = None, visible_device = "CPU")

    print("Name: ", model.name)
    print("Runner: ", model.runner)

    input = {
        "a": np.arange(20).reshape((4, 5))
    }

    print ("Input: ", input)

    model.infer(input)

asyncio.run(test())