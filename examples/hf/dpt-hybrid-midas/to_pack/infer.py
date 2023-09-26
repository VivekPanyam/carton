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

import torch
from transformers import DPTForDepthEstimation, DPTFeatureExtractor

class Model:
    def __init__(self):
        # Load the model we downloaded (notice the `local_files_only=True`)
        model_id = "Intel/dpt-hybrid-midas"
        model_revision = "fc1dad95a6337f3979a108e336932338130255a0"
        self.model = DPTForDepthEstimation.from_pretrained(
            model_id,
            revision = model_revision,
            cache_dir="./model",
            local_files_only=True,
        )

        self.feature_extractor = DPTFeatureExtractor.from_pretrained(
            model_id,
            revision = model_revision,
            cache_dir="./model",
            local_files_only=True,
        )

        if torch.cuda.is_available():
            self.model.to("cuda")

    def infer_with_tensors(self, tensors):
        image = tensors["image"]
        n, h, w, c = image.shape
        inputs = self.feature_extractor(images=image, return_tensors="pt")

        if torch.cuda.is_available():
            for k, v in inputs.items():
                inputs[k] = v.cuda()

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        # interpolate to original size
        # The input to interpolate is expected to be NCHW
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze(1).cpu().numpy()


        return {
            "depth": prediction
        }

def get_model():
    return Model()