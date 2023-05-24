import unittest
import numpy as np
import cartonml as carton
from cartonml.utils.transformers import pack_transformers_pipeline

class Test(unittest.IsolatedAsyncioTestCase):
    async def test_transformer_text_model(self):
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

        model_path = await pack_transformers_pipeline("fill-mask", "bert-base-uncased", postproc)
        print("Model path!", model_path)

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