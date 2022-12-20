from typing import Any, Optional
import carton
import asyncio
import numpy as np

# model = carton.load("somepath")
#
# a = np.array(4)
# b = np.array(4)
#
# # Let Carton know that this data is ready and won't be modified
# handle = model.seal({
#     "a": a,
#     "b": b
# })
#
#
# model.run(handle)


async def seal(model, input_queue: asyncio.Queue[Optional[Any]], output_queue: asyncio.Queue[Optional[Any]]):
    while True:
        # Get an item
        item = await input_queue.get()

        if item is None:
            # We're done
            input_queue.task_done()
            return
            
        # Seal the input
        result = model.seal(item)

        # Write the result out
        await output_queue.put(result)

        # Notify the queue that an item has been processed
        input_queue.task_done()

async def infer(model, input_queue: asyncio.Queue, output_queue: asyncio.Queue):
    while True:
        # Get an item
        item = await input_queue.get()

        # Run inference
        result = await model.run(item)

        # Write the result out
        await output_queue.put(result)

        # Notify the queue that an item has been processed
        input_queue.task_done()

async def pipeline_example():
    # Create a queue with max size one.
    # We only want at most one item being sealed while we're running inference
    # (This provides some backpressure and avoids the case where we're generating input data too quickly)
    # i.e. there's no point in sealing data faster than our model can run
    # This is especially important if the runner moves data to devices on seal. This can use up a lot of GPU memory
    # if we have too many things queued on GPU
    sealed_queue = asyncio.Queue(1)

    # Similarly, there's not really a reason to load data faster than it can be processed. It's safer to have a 
    # larger max size here, but it is important to have a max size so there's backpressure
    input_queue = asyncio.Queue(32)



    # Load examples and queue them for inference
    for example in range(50):
        await input_queue.put(example)

    await input_queue.join()
    await sealed_queue.join()

    