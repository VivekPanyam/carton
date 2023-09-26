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

const carton = require("../dist/node")

const express = require('express')
const app = express()

app.use(express.static('data'))

const loadTensorOrMisc = async (item) => {
    if (item.constructor.name === "MiscFileLoaderWrapper") {
        const stream = await item.get()
        return stream
    } else {
        const tensor = await item.get()
        return {
            buffer: tensor.buffer,
            shape: tensor.shape,
            dtype: tensor.dtype,
            stride: tensor.stride
        }
    }
}

const streamToBuffer = async (stream) => {
    const buffers = [];
    for await (const data of stream) {
        buffers.push(data);
    }

    return Buffer.concat(buffers);
}

async function run() {
    const server = await app.listen(0)
    const port = server.address().port;
    console.log("Running on ", port)
    console.time("get_model_info")
    let info = await carton.get_model_info(`http://localhost:${port}/test.carton`)
    // let info = await carton.get_model_info(`https://assets.carton.pub/manifest_sha256/f1cae5143a92a44e3a7e1d9c32a0245065dfb6e0f1e91c18b5e44356d081dfb5`)
    console.timeEnd("get_model_info")
    console.log(info.model_name)
    console.log(info.short_description)
    console.log(info.model_description)
    console.log(info.license)
    console.log(info.repository)
    console.log(info.homepage)
    console.log(info.required_platforms)
    console.log(info.inputs)
    console.log(info.outputs)

    for (const t of info.self_tests) {
        console.log(t.name)
        console.log(t.description)
        for (const [k, v] of t.inputs) {
            console.log(k, await loadTensorOrMisc(v))
        }
        for (const [k, v] of t.expected_out) {
            console.log(k, await loadTensorOrMisc(v))
        }
    }

    for (const t of info.examples) {
        console.log(t.name)
        console.log(t.description)
        for (const [k, v] of t.inputs) {
            console.log(k, await loadTensorOrMisc(v))
        }
        for (const [k, v] of t.sample_out) {
            console.log(k, await loadTensorOrMisc(v))
        }
    }


    console.log(info.runner)
    for (const [k, v] of info.misc_files) {
        console.log(k, await loadTensorOrMisc(v))
    }

    console.log(info.manifest_sha256)

    info.free()

    server.closeAllConnections()
    server.close()
}

run()
