// Copyright 2023 Vivek Panyam
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "../carton.h"

void infer_callback(CartonTensorMap *outputs, CartonStatus status, void *callback_arg)
{
    assert(status == CARTON_STATUS_SUCCESS);

    // Get the outputs
    CartonTensor *tokens_out;
    carton_tensormap_get_and_remove(outputs, "tokens", &tokens_out);

    CartonTensor *scores_out;
    carton_tensormap_get_and_remove(outputs, "scores", &scores_out);

    const char *token_str;
    uint64_t token_str_len;
    carton_tensor_get_string(tokens_out, 0, &token_str, &token_str_len);
    printf("Got output token: %.*s\n", (int)token_str_len, token_str);
    assert(strncmp(token_str, "day", token_str_len) == 0);

    float *scores_data;
    carton_tensor_data(scores_out, (void **)&scores_data);
    printf("Got output score: %f\n", scores_data[0]);

    carton_tensormap_destroy(outputs);
    carton_tensor_destroy(tokens_out);
    carton_tensor_destroy(scores_out);

    // The callback arg was the model
    carton_destroy((Carton *)callback_arg);

    exit(0);
}

void load_callback(Carton *model, CartonStatus status, void *callback_arg)
{
    assert(callback_arg == (void *)885);
    assert(status == CARTON_STATUS_SUCCESS);

    // Create a tensor
    CartonTensor *tensor;
    uint64_t dims[] = {1};
    carton_tensor_create(DATA_TYPE_STRING, dims, 1, &tensor);
    carton_tensor_set_string(tensor, 0, "Today is a good [MASK].");

    // Create an input map
    CartonTensorMap *tensors;
    carton_tensormap_create(&tensors);
    carton_tensormap_insert(tensors, "input", tensor);

    // Run inference
    carton_infer(model, tensors, infer_callback, model);
}

int main()
{
    carton_load("https://carton.pub/google-research/bert-base-uncased", load_callback, (void *)885);

    sleep(60);
}