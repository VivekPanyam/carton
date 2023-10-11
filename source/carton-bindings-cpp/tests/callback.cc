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

#include <cassert>
#include <iostream>
#include <unistd.h>

#include "../src/carton.hh"

using carton::Carton;
using carton::DataType;
using carton::Result;
using carton::Tensor;
using carton::TensorMap;

void infer_callback(Result<TensorMap> infer_result, void *arg)
{
    assert(arg == (void *)11);
    auto out = infer_result.get_or_throw();

    const auto tokens = out.get_and_remove("tokens");
    const auto scores = out.get_and_remove("scores");

    const auto scores_data = static_cast<const float *>(scores.data());

    std::cout << "Got output token: " << tokens.get_string(0) << std::endl;
    std::cout << "Got output scores: " << scores_data[0] << std::endl;

    exit(0);
}

void load_callback(Result<Carton> model_result, void *arg)
{
    assert(arg == (void *)42);
    auto model = model_result.get_or_throw();

    uint64_t shape[]{1};
    auto tensor = Tensor(DataType::kString, shape);
    tensor.set_string(0, "Today is a good [MASK].");

    std::unordered_map<std::string, Tensor> inputs;
    inputs.insert(std::make_pair("input", std::move(tensor)));

    // Run inference
    model.infer(std::move(inputs), infer_callback, (void *)11);
}

int main()
{
    // Load a model
    Carton::load("https://carton.pub/google-research/bert-base-uncased", load_callback, (void *)42);

    sleep(60);
}