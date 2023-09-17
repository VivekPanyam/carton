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