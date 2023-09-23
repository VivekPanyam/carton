import os
import asyncio
import logging
import numpy as np
import cartonml as carton

from cartonml import TensorSpec, Example, SelfTest
from cartonml.utils.hf import get_linked_files
from transformers import AutoProcessor, AutoModel


MODEL_DESCRIPTION = """
Bark is a transformer-based text-to-audio model created by [Suno](https://suno.ai). Bark can generate highly realistic, multilingual speech as well as other audio - including music, background noise and simple sound effects. The model can also produce nonverbal communications like laughing, sighing and crying.
"""

async def main():
    # Configure logging format
    FORMAT = '[%(asctime)s %(levelname)s %(name)s] %(filename)s:%(lineno)d %(message)s'
    logging.basicConfig(format=FORMAT)

    # We want trace messages to show up
    logging.getLogger().setLevel(5)

    # Download the models we need
    model_id = "suno/bark"
    model_revision = "2a3597836c810279492b1cbffa71b6056fba54cd"
    model = AutoModel.from_pretrained(
        model_id,
        revision = model_revision,
        cache_dir="./to_pack/model",
    )

    processor = AutoProcessor.from_pretrained(
        model_id,
        revision = model_revision,
        cache_dir="./to_pack/model",
    )

    # For a smaller output model file, we can let Carton know to pull large files from HF instead of storing
    # them in the output file. This can also lead to better caching behavior if the same files are used across
    # several cartons
    linked_files = get_linked_files(model_id, model_revision)

    # Pack the model with carton
    with open('./audio.wav', 'rb') as f:
        sample_audio = f.read()

    packed_model_path = await carton.pack(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "to_pack"),
        runner_name = "python",
        required_framework_version="=3.10",
        runner_opts = {
            "entrypoint_package": "infer",
            "entrypoint_fn": "get_model",
        },
        model_name = "bark",
        short_description = "Bark is a text to audio model",
        model_description = MODEL_DESCRIPTION,
        license = "MIT",
        homepage = "https://github.com/suno-ai/bark",
        inputs = [
            TensorSpec(
                name = "prompt",
                dtype = "string",
                shape = [],
                description = "The text to convert to audio"
            ),
        ],
        outputs = [
            TensorSpec(
                name = "audio",
                dtype = "float32",
                shape = ["duration"],
                description = "The audio as a 1D float32 array"
            ),
            TensorSpec(
                name = "sample_rate",
                dtype = "uint32",
                shape = [],
                description = "The sample rate of the audio"
            ),
        ],
        examples = [
            Example(
                inputs = dict(prompt = np.array("Hello, my name is Suno. And, uh — and I like pizza. [laughs] But I also have other interests such as playing tic tac toe.")),
                sample_out = dict(audio = sample_audio, sample_rate = np.array(24000, dtype=np.uint32))
            ),
        ],
        self_tests = [
            SelfTest(
                name = "quickstart",
                inputs = dict(prompt = np.array("Hello, my name is Suno. And, uh — and I like pizza. [laughs] But I also have other interests such as playing tic tac toe.")),
            )
        ],
        linked_files = linked_files,
    )

    print("Packed path", packed_model_path)

    # Testing
    model = await carton.load(packed_model_path)

    # out = await model.infer({
    #     "prompt": np.array("Hello, my name is Suno. And, uh — and I like pizza. [laughs] But I also have other interests such as playing tic tac toe.")
    # })

    # from scipy.io.wavfile import write as write_wav
    # write_wav("audio.wav", out["sample_rate"], out["audio"])

asyncio.run(main())