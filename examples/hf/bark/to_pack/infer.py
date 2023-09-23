import torch
import numpy as np
from transformers import AutoProcessor, AutoModel

class Model:
    def __init__(self):
        # Load the model we downloaded (notice the `local_files_only=True`)
        model_id = "suno/bark"
        model_revision = "2a3597836c810279492b1cbffa71b6056fba54cd"
        self.model = AutoModel.from_pretrained(
            model_id,
            revision = model_revision,
            cache_dir="./model",
            local_files_only=True,
        )

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            revision = model_revision,
            cache_dir="./model",
            local_files_only=True,
        )

        if torch.cuda.is_available():
            self.model.to("cuda")

    def infer_with_tensors(self, tensors):
        prompt = tensors["prompt"].tolist()
        
        inputs = self.processor(
            text=prompt,
            return_tensors="pt",
        )

        if torch.cuda.is_available():
            for k, v in inputs.items():
                inputs[k] = v.cuda()

        with torch.no_grad():
            audio = self.model.generate(**inputs).squeeze().cpu().numpy()


        return {
            "audio": audio,
            "sample_rate": np.array(self.model.generation_config.sample_rate, dtype=np.uint32)
        }

def get_model():
    return Model()