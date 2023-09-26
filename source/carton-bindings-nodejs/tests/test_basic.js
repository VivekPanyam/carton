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

const carton = require("..")

async function test(input) {
    const model = await carton.load("/tmp/somepath", {
        runner: "torchscript",
        visible_device: "CPU",
    })

    console.log("Name: ", model.name)
    console.log("Runner: ", model.runner)

    console.log("Input: ", input)

    let out = await model.infer(input)

    console.log("Out: ", out)
}


async function testStdlib() {
    const ndarray = require('@stdlib/ndarray');

    const input = {
        "a": ndarray.array(
            // Basically np.arange(20)
            new Float64Array([...Array(20).keys()]),
            {
                "shape": [4, 5]
            }
        )
    }

    await test(input)
}

testStdlib()

async function testndarray() {
    const ndarray = require('ndarray');

    const input = {
        "a": ndarray(
            // Basically np.arange(20)
            new Float64Array([...Array(20).keys()]),
            [4, 5]
        )
    }

    await test(input)
}

testndarray()