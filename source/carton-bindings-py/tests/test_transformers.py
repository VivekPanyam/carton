import unittest
import numpy as np
import cartonml as carton
from cartonml.utils.transformers import pack_transformers_pipeline

MODEL_PATH_CACHE = []

class Test(unittest.IsolatedAsyncioTestCase):
    async def test_transformer_text_model(self):
        model_path = await self._pack_model()
        await self._load_model(model_path)

    async def test_shrink(self):
        model_path = await self._pack_model()

        shrunk_path = await carton.shrink(
            model_path,
            {
                "097417381d6c7230bd9e3557456d726de6e83245ec8b24f529f60198a67b203a": ["https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin"]
            }
        )

        print("Shrunk", shrunk_path)
        await self._load_model(shrunk_path)

    async def _pack_model(self):
        """
        Pack a model and return the model path. This is cached so it can be called multiple times
        """
        def postproc(completions):
            if len(completions) > 0 and not isinstance(completions[0], list):
                # We need to wrap in another level
                completions = [completions]

            alltokens = []
            allscores = []
            for batch_item in completions:
                tokens = []
                scores = []
                for completion in batch_item:
                    tokens.append(completion['token_str'])
                    scores.append(completion['score'])

                alltokens.append(tokens)
                allscores.append(scores)

            return {
                "tokens": np.array(alltokens),
                "scores": np.array(allscores)
            }

        # Kinda hacky, but we can't use `self`` because the tests are isolated
        if len(MODEL_PATH_CACHE) == 0:
            MODEL_PATH_CACHE.append(await pack_transformers_pipeline("fill-mask", "bert-base-uncased", postproc))

        model_path = MODEL_PATH_CACHE[0]

        print("Model path!", model_path)

        return model_path

    async def _load_model(self, model_path):
        """
        Load and test a packed model
        """
        model = await carton.load(model_path)
        print("Loaded")

        out = await model.infer({
            "input_sequences": np.array([
                "Today is a good [MASK].",
                "The [MASK] went around the track.",
            ])
        })

        print(out)

        scores = out["scores"]
        tokens = out["tokens"]

        # Pick the tokens with the highest score for each input
        selections = np.take_along_axis(tokens, scores.argmax(axis=1, keepdims=True), axis=1).flatten()

        self.assertEqual(selections[0], "day")
        self.assertEqual(selections[1], "car")


if __name__ == "__main__":
    unittest.main()