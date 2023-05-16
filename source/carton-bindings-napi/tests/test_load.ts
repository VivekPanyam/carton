import test from 'ava'

import { load, loadUnpacked, pack, LoadUnpackedArgs, Carton, RunnerInfo } from '../index.cjs'
import fs from 'fs/promises';
import path from 'path';
import os from 'os';

import fetch from 'node-fetch';
import ndarray from '@stdlib/ndarray';

const MODEL_DESCRIPTION = `
Some description of the model that can be as detailed as you want it to be

This can span multiple lines. It can use markdown, but shouldn't use arbitrary HTML
(which will usually be treated as text instead of HTML when this is rendered)

You can use images and links, but the paths must either be complete https
URLs (e.g. https://example.com/image.png) or must reference a file in the misc folder (e.g \`@misc/file.png\`).
See the \`misc_files\` section of this document for more details

![Model Architecture](@misc/model_architecture.png)
`


function convertTensors(tensors) {
  // TODO: this may be a bit brittle across libraries so refactor if needed

  // Get the buffer, shape, stride, and dtype for each input tensor
  const nativeTensors = {};
  for (const key in tensors) {
    const tensor = tensors[key];

    // Should support `ndarray` and `@stdlib/ndarray`
    let buffer = Buffer.from(tensor.data.buffer)
    let shape = tensor.shape
    var stride = tensor.stride || tensor.strides
    let dtype = tensor.dtype

    if (shape.length == 0) {
      stride = []
    }

    nativeTensors[key] = { buffer, shape, dtype, stride }
  }

  // Run the model
  return nativeTensors
}

const run_load_test = async (t, load_fn: (arg: LoadUnpackedArgs) => Promise<Carton>) => {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'carton-tests-'))

  await fs.writeFile(path.join(dir, "requirements.txt"), "numpy")
  await fs.writeFile(path.join(dir, "main.py"), `
import sys
import numpy as np

class Model:
    def __init__(self):
        pass

    def infer_with_tensors(self, tensors):
        return {
            "out": np.array([sys.version_info.major, sys.version_info.minor], dtype=np.uint32)
        }

def get_model():
    return Model()
  `)

  const model = await load_fn({
    path: dir,
    runnerName: "python",
    requiredFrameworkVersion: "=3.11",
    runnerOpts: {
      "entrypoint_package": "main",
      "entrypoint_fn": "get_model",
    },
    modelName: "Test Model",
    shortDescription: "A short description that should be less than or equal to 100 characters.",
    modelDescription: MODEL_DESCRIPTION,
    requiredPlatforms: [
      "x86_64-unknown-linux-gnu",
      "aarch64-unknown-linux-gnu",
      "x86_64-apple-darwin",
      "aarch64-apple-darwin",
    ],
    inputs: [
      {
        name: "a",
        dtype: "float32",
        shape: null,
      },
      {
        name: "b",
        dtype: "float64",
        shape: "some_symbol",
      },
      {
        name: "c",
        dtype: "string",
        shape: [null],
      },
      {
        name: "d",
        dtype: "int8",
        shape: ["some_symbol"],
      },
      {
        name: "e",
        dtype: "int16",
        shape: [],
      },
      {
        name: "f",
        dtype: "int32",
        shape: ["batch_size", 3, 512, 512],
      },
      {
        name: "g",
        dtype: "int64",
        shape: [null, null, null],
      },
    ],
    outputs: [
      {
        name: "y",
        dtype: "uint32",
        shape: [2, 3, 512, 512],
        description: "Some description of the output"
      }
    ],
    selfTests: [
      {
        name: "a_self_test",
        description: "A self test",
        inputs: convertTensors({
          // a: np.ones(5, dtype = np.float32)
          a: ndarray.array(new Float32Array([1, 1, 1, 1, 1]), { "shape": [5] })
        }),
        expectedOut: convertTensors({
          // y: np.ones(5, dtype = np.uint32)
          y: ndarray.array(new Uint32Array([1, 1, 1, 1, 1]), { "shape": [5] })
        })
      }
    ],
    examples: [
      {
        name: "an_optional_name",
        description: "An example of an image input",
        inputs: { f: Buffer.from(t.context.input_image) },
        sampleOut: convertTensors({
          // y: np.array(5)
          y: ndarray.ndarray("uint32", new Uint32Array([5]), [], [0], 0, 'row-major')
        })
      }
    ],
    miscFiles: {
      "model_architecture.png": Buffer.from(t.context.model_architecture)
    },
    visibleDevice: "CPU"
  })

  let out = await model.infer({})
  out = out["out"]

  // TODO: multiply each item of `out.stride` by byte length since we're passing in an untyped buffer
  var converted = ndarray.ndarray(out.dtype, out.buffer, out.shape, [4], 0, 'row-major')

  const major = converted.get(0)
  const minor = converted.get(1)

  t.is(major, 3)
  t.is(minor, 11)

  const info = await model.getInfo()
  
  // Extract the example image from the model
  const exampleImage = await info.examples[0].inputs["f"].read()
  t.assert(exampleImage.equals(Buffer.from(t.context.input_image)))

  const exampleSampleOut = await info.examples[0].sampleOut["y"].get()
  t.deepEqual(exampleSampleOut.shape, [])
  t.deepEqual(exampleSampleOut.stride, [])
  t.is(exampleSampleOut.dtype, "uint32")
  t.assert(Buffer.from(new Uint32Array([5]).buffer).equals(exampleSampleOut.buffer))

  const selfTestOut = await info.selfTests[0].expectedOut["y"].get()
  t.deepEqual(selfTestOut.shape, [5])
  t.deepEqual(selfTestOut.stride, [1])
  t.is(selfTestOut.dtype, "uint32")
  t.assert(Buffer.from(new Uint32Array([1, 1, 1, 1, 1]).buffer).equals(selfTestOut.buffer))

  // Extract the model architecture image
  const modelArchReadback = await info.miscFiles["model_architecture.png"].read()
  t.assert(Buffer.from(t.context.model_architecture).equals(modelArchReadback))
  
}

test.before(async t => {
  t.context = {}
  {
    const res = await fetch("https://www.cis.upenn.edu/~jshi/ped_html/images/PennPed00015_1.png");
    // @ts-ignore-nextline
    t.context.input_image = await res.arrayBuffer()
  }
  
  {
    const res = await fetch("https://raw.githubusercontent.com/openai/whisper/main/approach.png")
    // @ts-ignore-nextline
    t.context.model_architecture = await res.arrayBuffer()
  }
})

test('test loadUnpacked', async t => {
  await run_load_test(t, loadUnpacked)
})

test('test pack followed by load', async t => {
  await run_load_test(t, async ({ visibleDevice, ...packArgs }) => {
    const model_path = await pack(packArgs)

    return await load({ urlOrPath: model_path, visibleDevice })
  })
})