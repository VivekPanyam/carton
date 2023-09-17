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